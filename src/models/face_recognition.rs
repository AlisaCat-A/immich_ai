use anyhow::Result;
use ndarray::Array4;
use ort::{Session, Value};
use image::{DynamicImage, GenericImageView, imageops};
use serde::Serialize;

// Face Recognition (ArcFace) usually outputs a 512-dim embedding
#[derive(Debug, Clone, Serialize)]
pub struct FaceRecognitionOutput {
    pub embedding: Vec<f32>,
}

pub struct FaceRecognitionModel {
    session: Session,
    input_size: (u32, u32), // usually 112x112 for ArcFace
}

impl FaceRecognitionModel {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            input_size: (112, 112),
        })
    }

    pub fn process(&self, face_image: &DynamicImage) -> Result<FaceRecognitionOutput> {
        // Preprocess: Resize to 112x112, Normalize
        let (w, h) = self.input_size;
        let resized = face_image.resize_exact(w, h, imageops::FilterType::Triangle);

        let mut input_tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));

        for (x, y, pixel) in resized.pixels() {
            let rgb = pixel.0;
            input_tensor[[0, 0, y as usize, x as usize]] = rgb[0] as f32;
            input_tensor[[0, 1, y as usize, x as usize]] = rgb[1] as f32;
            input_tensor[[0, 2, y as usize, x as usize]] = rgb[2] as f32;
        }

        // ArcFace Normalization: usually (x - 127.5) / 128.0
        input_tensor.mapv_inplace(|v| (v - 127.5) / 128.0);

        let input_value = Value::from_array(input_tensor)?;
        let outputs = self.session.run(ort::inputs!["input.1" => input_value]?)?;

        // Output is usually "embedding" (1x512)
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let embedding = output_tensor.view().to_slice().unwrap().to_vec();

        Ok(FaceRecognitionOutput {
            embedding,
        })
    }
}
