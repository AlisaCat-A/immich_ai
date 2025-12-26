use crate::models::{ModelConfig, ModelType, SafeModelManager};
use std::path::PathBuf;

pub async fn register_default_models(model_manager: &mut SafeModelManager) {
    // 定义基础路径
    let models_dir = PathBuf::from("models");
    let tokenizer_path = models_dir.join("clip_cn_tokenizer.json").to_string_lossy().to_string();
    let max_length = Some(52);

    // --- Chinese CLIP Models ---
    // 注册 ViT-B-16
    model_manager.register_model(
        "ViT-B-16__openai".to_string(),
        ModelConfig {
            model_type: ModelType::Image,
            model_path: models_dir.join("ViT-B-16.img.fp32.onnx").to_string_lossy().to_string(),
            tokenizer_path: None,
            image_size: Some((224, 224)),
            max_length,
        }
    ).await;

    model_manager.register_model(
        "ViT-B-16__openai_text".to_string(),
        ModelConfig {
            model_type: ModelType::Text,
            model_path: models_dir.join("ViT-B-16.txt.fp32.onnx").to_string_lossy().to_string(),
            tokenizer_path: Some(tokenizer_path.clone()),
            image_size: None,
            max_length,
        }
    ).await;

    // 注册 ViT-L-14
    model_manager.register_model(
        "ViT-L-14__openai".to_string(),
        ModelConfig {
            model_type: ModelType::Image,
            model_path: models_dir.join("ViT-L-14.img.fp32.onnx").to_string_lossy().to_string(),
            tokenizer_path: None,
            image_size: Some((224, 224)),
            max_length,
        }
    ).await;

    model_manager.register_model(
        "ViT-L-14__openai_text".to_string(),
        ModelConfig {
            model_type: ModelType::Text,
            model_path: models_dir.join("ViT-L-14.txt.fp32.onnx").to_string_lossy().to_string(),
            tokenizer_path: Some(tokenizer_path.clone()),
            image_size: None,
            max_length,
        }
    ).await;

    // 注册 ViT-L-14-336
    model_manager.register_model(
        "ViT-L-14-336__openai".to_string(),
        ModelConfig {
            model_type: ModelType::Image,
            model_path: models_dir.join("ViT-L-14-336.img.fp32.onnx").to_string_lossy().to_string(),
            tokenizer_path: None,
            image_size: Some((336, 336)),
            max_length,
        }
    ).await;

    model_manager.register_model(
        "ViT-L-14-336__openai_text".to_string(),
        ModelConfig {
            model_type: ModelType::Text,
            model_path: models_dir.join("ViT-L-14-336.txt.fp32.onnx").to_string_lossy().to_string(),
            tokenizer_path: Some(tokenizer_path.clone()),
            image_size: None,
            max_length,
        }
    ).await;

    // --- Face Detection (Buffalo L / SCRFD) ---
    model_manager.register_model(
        "buffalo_l".to_string(), // Detection is requested as "buffalo_l" with task "detection"
        ModelConfig {
            model_type: ModelType::FaceDetection,
            model_path: models_dir.join("buffalo_l").join("det_10g.onnx").to_string_lossy().to_string(),
            tokenizer_path: None,
            image_size: None,
            max_length: None,
        }
    ).await;
    // Also register "antelopev2" as it's common? Just buffalo_l for now as requested.

    // --- Face Recognition (Buffalo L / ArcFace) ---
    // Note: Immich might use the same model name "buffalo_l" but different task/type.
    // Our registry key is just the name. We might need a suffix to distinguish if the name is identical.
    // Immich request: task=facial-recognition, type=detection, modelName=buffalo_l
    // Immich request: task=facial-recognition, type=recognition, modelName=buffalo_l

    // In `main.rs`, we can handle this by appending suffix like `_recognition`.
    model_manager.register_model(
        "buffalo_l_recognition".to_string(),
        ModelConfig {
            model_type: ModelType::FaceRecognition,
            model_path: models_dir.join("buffalo_l").join("w600k_r50.onnx").to_string_lossy().to_string(),
            tokenizer_path: None,
            image_size: None,
            max_length: None,
        }
    ).await;

    // --- Multilingual CLIP (SigLIP) ---
    // nllb-clip-large-siglip__v1
    let siglip_path = models_dir.join("nllb-clip-large-siglip__v1");
    let siglip_tokenizer = siglip_path.join("tokenizer.json").to_string_lossy().to_string(); // Assuming this path

    model_manager.register_model(
        "nllb-clip-large-siglip__v1".to_string(),
        ModelConfig {
            model_type: ModelType::SiglipVision,
            model_path: models_dir.join("nllb-clip-large-siglip__v1").join("vision.onnx").to_string_lossy().to_string(), // Verify filename
            tokenizer_path: None,
            image_size: Some((224, 224)),
            max_length: None,
        }
    ).await;

    model_manager.register_model(
        "nllb-clip-large-siglip__v1_text".to_string(),
        ModelConfig {
            model_type: ModelType::SiglipText,
            model_path: models_dir.join("nllb-clip-large-siglip__v1").join("text.onnx").to_string_lossy().to_string(), // Verify filename
            tokenizer_path: Some(siglip_tokenizer),
            image_size: None,
            max_length: None,
        }
    ).await;
}
