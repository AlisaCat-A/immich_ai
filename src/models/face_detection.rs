use anyhow::{Result, Context};
use ndarray::Array4;
use ort::{Session, Value};
use image::{DynamicImage, GenericImageView, imageops};
use serde::Serialize;
use std::cmp::Ordering;

// Based on typical SCRFD / Buffalo_L outputs
#[derive(Debug, Clone, Serialize)]
pub struct FaceDetectionOutput {
    pub boxes: Vec<[f32; 4]>,     // x1, y1, x2, y2
    pub scores: Vec<f32>,
    pub landmarks: Vec<[f32; 10]>, // 5 points (x,y) flattened
}

pub struct FaceDetectionModel {
    session: Session,
    input_size: (u32, u32), // default 640x640 for detection
    strides: Vec<u32>,
}

impl FaceDetectionModel {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            input_size: (640, 640),
            strides: vec![8, 16, 32],
        })
    }

    pub fn process(&self, image_data: &[u8]) -> Result<FaceDetectionOutput> {
        let img = image::load_from_memory(image_data)
            .context("Failed to load image from memory")?;

        let (input_tensor, ratio, _dw, _dh) = self.preprocess(&img)?;

        let input_value = Value::from_array(input_tensor)?;

        // Run inference
        let outputs = self.session.run(ort::inputs!["input.1" => input_value]?)?;

        // Decode outputs
        let mut proposals = Vec::new();

        // SCRFD typically outputs 3 sets of tensors per stride (score, bbox, kps).
        // Total outputs = strides.len() * 3 = 9.
        // The order in `outputs` depends on the model graph. Usually it's grouped by stride or type.
        // Often: [score8, bbox8, kps8, score16, bbox16, kps16, score32, bbox32, kps32]
        // or: [score8, score16, score32, bbox8, bbox16, bbox32, kps8, kps16, kps32]
        // Without inspecting the model, let's assume the standard InsightFace export order:
        // usually sorted by name.
        // Let's assume the flattened order: stride 8 (3 tensors), stride 16 (3), stride 32 (3).

        if outputs.len() < 9 {
             // Fallback or error if model doesn't match expectation
             return Ok(FaceDetectionOutput { boxes: vec![], scores: vec![], landmarks: vec![] });
        }

        // We interpret outputs in groups of 3
        for (i, stride) in self.strides.iter().enumerate() {
            let score_idx = i * 3;
            let bbox_idx = i * 3 + 1;
            let kps_idx = i * 3 + 2;

            let score_tensor = outputs[score_idx].try_extract_tensor::<f32>()?;
            let bbox_tensor = outputs[bbox_idx].try_extract_tensor::<f32>()?;
            let kps_tensor = outputs[kps_idx].try_extract_tensor::<f32>()?;

            // Generate anchors for this stride
            let feat_h = self.input_size.1 / stride;
            let feat_w = self.input_size.0 / stride;

            // Check if tensor shapes match expectation (Batch, Channels, H, W) or (Batch, H, W, Channels)
            // SCRFD usually (1, C, H, W)
            // Score: (1, 1, H, W)
            // BBox: (1, 4, H, W)
            // Kps: (1, 10, H, W)

            // Iterate over H, W
            let score_view = score_tensor.view().into_dimensionality::<ndarray::Ix4>()?;
            let bbox_view = bbox_tensor.view().into_dimensionality::<ndarray::Ix4>()?;
            let kps_view = kps_tensor.view().into_dimensionality::<ndarray::Ix4>()?;

            let threshold = 0.5; // Confidence threshold

            for y in 0..feat_h {
                for x in 0..feat_w {
                    let score = score_view[[0, 0, y as usize, x as usize]];
                    if score < threshold {
                        continue;
                    }

                    // Anchor center
                    let anchor_x = (x * stride) as f32;
                    let anchor_y = (y * stride) as f32;

                    // Decode BBox (distance from anchor: l, t, r, b)
                    // bbox_tensor has shape (1, 4, H, W)
                    let l = bbox_view[[0, 0, y as usize, x as usize]] * (*stride as f32);
                    let t = bbox_view[[0, 1, y as usize, x as usize]] * (*stride as f32);
                    let r = bbox_view[[0, 2, y as usize, x as usize]] * (*stride as f32);
                    let b = bbox_view[[0, 3, y as usize, x as usize]] * (*stride as f32);

                    let x1 = anchor_x - l;
                    let y1 = anchor_y - t;
                    let x2 = anchor_x + r;
                    let y2 = anchor_y + b;

                    // Decode Landmarks (5 points * 2 coords)
                    // kps_tensor has shape (1, 10, H, W)
                    let mut landmarks = [0.0; 10];
                    for k in 0..5 {
                        landmarks[k*2] = anchor_x + kps_view[[0, k*2, y as usize, x as usize]] * (*stride as f32);
                        landmarks[k*2+1] = anchor_y + kps_view[[0, k*2+1, y as usize, x as usize]] * (*stride as f32);
                    }

                    proposals.push(Proposal {
                        box_x1: x1, box_y1: y1, box_x2: x2, box_y2: y2,
                        score,
                        landmarks,
                    });
                }
            }
        }

        let keep_indices = nms(&proposals, 0.4);

        let mut final_boxes = Vec::new();
        let mut final_scores = Vec::new();
        let mut final_landmarks = Vec::new();

        for idx in keep_indices {
            let p = &proposals[idx];
            // Rescale back to original image coordinates
            final_boxes.push([
                p.box_x1 / ratio,
                p.box_y1 / ratio,
                p.box_x2 / ratio,
                p.box_y2 / ratio
            ]);
            final_scores.push(p.score);

            let mut scaled_lm = [0.0; 10];
            for k in 0..10 {
                scaled_lm[k] = p.landmarks[k] / ratio;
            }
            final_landmarks.push(scaled_lm);
        }

        Ok(FaceDetectionOutput {
            boxes: final_boxes,
            scores: final_scores,
            landmarks: final_landmarks,
        })
    }

    fn preprocess(&self, img: &DynamicImage) -> Result<(Array4<f32>, f32, f32, f32)> {
        let (w, h) = img.dimensions();
        let (target_w, target_h) = self.input_size;

        let r = (target_w as f32 / w as f32).min(target_h as f32 / h as f32);

        let new_w = (w as f32 * r) as u32;
        let new_h = (h as f32 * r) as u32;
        let resized = img.resize_exact(new_w, new_h, imageops::FilterType::Triangle);

        let mut input_tensor = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));

        for (x, y, pixel) in resized.pixels() {
            let rgb = pixel.0;
            input_tensor[[0, 0, y as usize, x as usize]] = rgb[0] as f32;
            input_tensor[[0, 1, y as usize, x as usize]] = rgb[1] as f32;
            input_tensor[[0, 2, y as usize, x as usize]] = rgb[2] as f32;
        }

        input_tensor.mapv_inplace(|v| (v - 127.5) / 128.0);

        Ok((input_tensor, r, 0.0, 0.0))
    }
}

struct Proposal {
    box_x1: f32,
    box_y1: f32,
    box_x2: f32,
    box_y2: f32,
    score: f32,
    landmarks: [f32; 10],
}

fn nms(proposals: &[Proposal], iou_threshold: f32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..proposals.len()).collect();
    // Sort descending by score
    indices.sort_by(|&a, &b| proposals[b].score.partial_cmp(&proposals[a].score).unwrap_or(Ordering::Equal));

    let mut keep = Vec::new();
    let mut suppressed = vec![false; proposals.len()];

    for i in 0..indices.len() {
        let idx_i = indices[i];
        if suppressed[idx_i] {
            continue;
        }
        keep.push(idx_i);

        for j in (i + 1)..indices.len() {
            let idx_j = indices[j];
            if suppressed[idx_j] {
                continue;
            }

            let iou = compute_iou(&proposals[idx_i], &proposals[idx_j]);
            if iou > iou_threshold {
                suppressed[idx_j] = true;
            }
        }
    }
    keep
}

fn compute_iou(a: &Proposal, b: &Proposal) -> f32 {
    let x1 = a.box_x1.max(b.box_x1);
    let y1 = a.box_y1.max(b.box_y1);
    let x2 = a.box_x2.min(b.box_x2);
    let y2 = a.box_y2.min(b.box_y2);

    let w = (x2 - x1).max(0.0);
    let h = (y2 - y1).max(0.0);
    let inter = w * h;

    let area_a = (a.box_x2 - a.box_x1) * (a.box_y2 - a.box_y1);
    let area_b = (b.box_x2 - b.box_x1) * (b.box_y2 - b.box_y1);

    if area_a + area_b - inter <= 0.0 {
        return 0.0;
    }

    inter / (area_a + area_b - inter)
}
