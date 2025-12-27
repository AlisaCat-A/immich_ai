use anyhow::Result;
use ndarray::{Array2, Array4};
use ort::{Session, Value};
use image::{GenericImageView, imageops};
use tokenizers::Tokenizer;

pub struct SiglipModel {
    vision_session: Option<Session>,
    text_session: Option<Session>,
    tokenizer: Option<Tokenizer>,
    image_size: (u32, u32),
}

impl SiglipModel {
    pub fn new_vision(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path)?;

        Ok(Self {
            vision_session: Some(session),
            text_session: None,
            tokenizer: None,
            image_size: (224, 224), // Default SigLIP size
        })
    }

    pub fn new_text(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            vision_session: None,
            text_session: Some(session),
            tokenizer: Some(tokenizer),
            image_size: (0, 0),
        })
    }

    pub fn process_image(&self, image_data: &[u8]) -> Result<Vec<f32>> {
        let session = self.vision_session.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vision model not loaded"))?;

        let img = image::load_from_memory(image_data)?;
        let (w, h) = self.image_size;
        let resized = img.resize_exact(w, h, imageops::FilterType::Triangle);

        let mut input_tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));

        let mean = [0.48145466, 0.4578275, 0.40821073];
        let std = [0.26862954, 0.26130258, 0.27577711];

        for (x, y, pixel) in resized.pixels() {
            let rgb = pixel.0;
            for c in 0..3 {
                let v = (rgb[c] as f32 / 255.0 - mean[c]) / std[c];
                input_tensor[[0, c, y as usize, x as usize]] = v;
            }
        }

        let input_value = Value::from_array(input_tensor)?;
        let outputs = session.run(ort::inputs!["pixel_values" => input_value]?)?;

        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;

        Ok(output_tensor.view().to_slice().unwrap().to_vec())
    }

    pub fn process_text(&self, text: &str) -> Result<Vec<f32>> {
        let session = self.text_session.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Text model not loaded"))?;
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;

        let encoding = tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("Encoding error: {}", e))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();

        let batch_size = 1;
        let seq_len = ids.len();

        let input_ids = Array2::from_shape_vec((batch_size, seq_len), ids)?;
        let input_mask = Array2::from_shape_vec((batch_size, seq_len), attention_mask)?;

        let input_ids_val = Value::from_array(input_ids)?;
        let input_mask_val = Value::from_array(input_mask)?;

        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_val,
            "attention_mask" => input_mask_val
        ]?)?;

        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        Ok(output_tensor.view().to_slice().unwrap().to_vec())
    }
}
