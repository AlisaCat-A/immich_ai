use actix_multipart::form::bytes::Bytes;
use actix_multipart::form::text::Text;
use actix_multipart::form::MultipartForm;
use actix_web::{web, App, HttpServer, HttpResponse};
use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;
use log::{info, error};
use actix_web::middleware::Logger;
use crate::models::SafeModelManager;
use std::collections::HashMap;
use std::env;
use std::panic;
use image::{DynamicImage, GenericImageView};

mod models;
mod model_registry;
mod image_utils;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Model, ModelInput, ModelOutput, SafeModelManager, face_detection::FaceDetectionOutput};
    use std::sync::Arc;

    // Mock Models
    struct MockClipModel;
    impl Model for MockClipModel {
        fn process(&self, _input: &ModelInput) -> Result<ModelOutput, anyhow::Error> {
            Ok(ModelOutput::Clip(vec![0.1, 0.2, 0.3]))
        }
    }

    struct MockFaceDetectionModel;
    impl Model for MockFaceDetectionModel {
        fn process(&self, _input: &ModelInput) -> Result<ModelOutput, anyhow::Error> {
            Ok(ModelOutput::FaceDetection(FaceDetectionOutput {
                boxes: vec![[0.0, 0.0, 100.0, 100.0]],
                scores: vec![0.99],
                landmarks: vec![],
            }))
        }
    }

    struct MockFaceRecognitionModel;
    impl Model for MockFaceRecognitionModel {
        fn process(&self, _input: &ModelInput) -> Result<ModelOutput, anyhow::Error> {
            Ok(ModelOutput::FaceRecognition(crate::models::face_recognition::FaceRecognitionOutput {
                 embedding: vec![0.5, 0.5]
            }))
        }
    }

    #[actix_web::test]
    async fn test_clip_prediction() {
        let model_manager = SafeModelManager::new();
        model_manager.register_model_instance("ViT-B-16__openai".to_string(), Arc::new(MockClipModel)).await;
        model_manager.register_model_instance("ViT-B-16__openai_text".to_string(), Arc::new(MockClipModel)).await;

        let state = web::Data::new(AppState { model_manager });

        let model = state.model_manager.get_model("ViT-B-16__openai").await.unwrap();
        let output = model.process(&models::ModelInput::Text("test".to_string())).unwrap();
        if let models::ModelOutput::Clip(features) = output {
            assert_eq!(features, vec![0.1, 0.2, 0.3]);
        } else {
            panic!("Wrong output type");
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 3003)]
    port: u16,
}

// --- Immich-compatible Schema Definitions ---

#[derive(Debug, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
enum ModelTask {
    FacialRecognition,
    Clip,
}

#[derive(Debug, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
enum ModelTypeEnum {
    Detection,
    Recognition,
    Textual,
    Visual,
}

#[derive(Debug, Deserialize)]
struct PipelineEntry {
    #[serde(rename = "modelName")]
    model_name: String,
    #[serde(default)]
    options: Value,
}

// Mapping: Task -> { Type -> Entry }
type PipelineRequest = HashMap<ModelTask, HashMap<ModelTypeEnum, PipelineEntry>>;

#[derive(Debug, MultipartForm)]
struct PredictForm {
    entries: Text<String>,
    text: Option<Text<String>>,
    image: Option<Bytes>,
}

struct AppState {
    model_manager: SafeModelManager,
}

async fn predict(
    state: web::Data<AppState>,
    MultipartForm(form): MultipartForm<PredictForm>,
) -> Result<HttpResponse, actix_web::Error> {
    // 1. Parse the request JSON
    let entries: PipelineRequest = serde_json::from_str(&form.entries.into_inner())
        .map_err(|e| {
             error!("Failed to parse request JSON: {}", e);
             actix_web::error::ErrorBadRequest(format!("Invalid request format: {}", e))
        })?;

    let mut response_data = serde_json::Map::new();

    // 2. Handle 'clip' task
    if let Some(clip_task) = entries.get(&ModelTask::Clip) {
        // Handle Visual (Image)
        if let Some(visual_entry) = clip_task.get(&ModelTypeEnum::Visual) {
            if let Some(image_data) = &form.image {
                let model = state.model_manager.get_model(&visual_entry.model_name).await
                    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                let image_bytes = image_data.data.to_vec();
                let output = web::block(move || model.process(&models::ModelInput::Image(image_bytes)))
                    .await
                    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?
                    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                 if let models::ModelOutput::Clip(features) = output {
                    response_data.insert("clip".to_string(), json!(features));
                }
            }
        }
        
        // Handle Textual (Text)
        if let Some(text_entry) = clip_task.get(&ModelTypeEnum::Textual) {
            if let Some(text) = &form.text {
                 // Try standard suffix
                let model_name = format!("{}_text", text_entry.model_name);
                let model = state.model_manager.get_model(&model_name).await
                    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                let text_val = text.to_string();
                let output = web::block(move || model.process(&models::ModelInput::Text(text_val)))
                    .await
                    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?
                    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                if let models::ModelOutput::Clip(features) = output {
                    response_data.insert("clip".to_string(), json!(features));
                }
            }
        }
    }

    // 3. Handle 'facial-recognition' task
    if let Some(face_task) = entries.get(&ModelTask::FacialRecognition) {
        if let Some(detection_entry) = face_task.get(&ModelTypeEnum::Detection) {
             if let Some(image_data) = &form.image {
                // 3a. Run Detection
                let detection_model_name = detection_entry.model_name.clone();
                let det_model = state.model_manager.get_model(&detection_model_name).await
                    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                let image_bytes = image_data.data.to_vec();
                let det_output = web::block(move || det_model.process(&models::ModelInput::Image(image_bytes)))
                    .await
                    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?
                    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                let faces_data = if let models::ModelOutput::FaceDetection(output) = det_output {
                    // Load original image for cropping
                    let original_img = image::load_from_memory(&image_data.data)
                        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                    // Inject Image Dimensions
                    response_data.insert("imageHeight".to_string(), json!(original_img.height()));
                    response_data.insert("imageWidth".to_string(), json!(original_img.width()));

                    if let Some(recognition_entry) = face_task.get(&ModelTypeEnum::Recognition) {
                        let rec_model_name = format!("{}_recognition", recognition_entry.model_name);
                         let rec_model = state.model_manager.get_model(&rec_model_name).await
                            .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                        let mut detected_faces = Vec::new();

                        // Iterate over detected boxes
                        for (i, bbox) in output.boxes.iter().enumerate() {
                            let score = output.scores.get(i).cloned().unwrap_or(0.0);

                            // Align face using landmarks
                            let landmarks = output.landmarks.get(i).ok_or_else(|| actix_web::error::ErrorInternalServerError("Missing landmarks"))?;
                            let face_crop = image_utils::align_face(&original_img, landmarks);

                            // Pass the crop directly to the recognition model (optimized path)
                            // Note: process() is blocking, so we should wrap this too if possible,
                            // but since we are iterating, we might want to do it differently or just accept it's on a thread if we wrapped the whole block.
                            // However, we are currently in the async handler main body (after awaiting det_output).
                            // We should wrap the whole recognition loop or per-face.
                            // Wrapping per-face adds overhead.
                            // Better: move the whole logic into a web::block.

                            let rec_model_clone = rec_model.clone();
                            let rec_output = web::block(move || {
                                rec_model_clone.process(&models::ModelInput::ImageRaw(face_crop))
                            }).await
                            .map_err(|e| actix_web::error::ErrorInternalServerError(e))?
                            .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                            if let models::ModelOutput::FaceRecognition(rec_result) = rec_output {
                                detected_faces.push(json!({
                                    "boundingBox": {
                                        "x1": bbox[0] as i32,
                                        "y1": bbox[1] as i32,
                                        "x2": bbox[2] as i32,
                                        "y2": bbox[3] as i32
                                    },
                                    "embedding": rec_result.embedding,
                                    "score": score
                                }));
                            }
                        }

                        json!(detected_faces)
                    } else {
                         // Only detection requested
                         json!(output)
                    }
                } else {
                    json!([])
                };

                response_data.insert("facial-recognition".to_string(), faces_data);
            }
        }
    }

    Ok(HttpResponse::Ok().json(response_data))
}

// 空闲检查任务
async fn idle_shutdown_task(state: web::Data<AppState>, model_ttl: u64, poll_interval: u64) {
    loop {
        // 清理过期模型
        state.model_manager.cleanup_expired_models(Duration::from_secs(model_ttl)).await;
        tokio::time::sleep(tokio::time::Duration::from_secs(poll_interval)).await;
    }
}

async fn preload_models(model_manager: &SafeModelManager) {
    // Check for MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL
    if let Ok(model_name) = env::var("MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL") {
        info!("Preloading text model: {}", model_name);
        let text_model_name = format!("{}_text", model_name);

        match model_manager.get_model(&text_model_name).await {
            Ok(_) => info!("Successfully preloaded {}", text_model_name),
            Err(e) => error!("Failed to preload {}: {}", text_model_name, e),
        }
    }

    // Check for MACHINE_LEARNING_PRELOAD__CLIP__VISUAL
    if let Ok(model_name) = env::var("MACHINE_LEARNING_PRELOAD__CLIP__VISUAL") {
        info!("Preloading visual model: {}", model_name);
        match model_manager.get_model(&model_name).await {
            Ok(_) => info!("Successfully preloaded {}", model_name),
            Err(e) => error!("Failed to preload {}: {}", model_name, e),
        }
    }

    // Check for MACHINE_LEARNING_PRELOAD__FACIAL_RECOGNITION__DETECTION
    if let Ok(model_name) = env::var("MACHINE_LEARNING_PRELOAD__FACIAL_RECOGNITION__DETECTION") {
        info!("Preloading face detection model: {}", model_name);
        match model_manager.get_model(&model_name).await {
            Ok(_) => info!("Successfully preloaded {}", model_name),
            Err(e) => error!("Failed to preload {}: {}", model_name, e),
        }
    }

    // Check for MACHINE_LEARNING_PRELOAD__FACIAL_RECOGNITION__RECOGNITION
    if let Ok(model_name) = env::var("MACHINE_LEARNING_PRELOAD__FACIAL_RECOGNITION__RECOGNITION") {
        info!("Preloading face recognition model: {}", model_name);
        let rec_model_name = format!("{}_recognition", model_name);
        match model_manager.get_model(&rec_model_name).await {
            Ok(_) => info!("Successfully preloaded {}", rec_model_name),
            Err(e) => error!("Failed to preload {}: {}", rec_model_name, e),
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    panic::set_hook(Box::new(|panic_info| {
        error!("CRASH: Application panicked: {:?}", panic_info);
    }));

    if env::var("RUST_LOG").is_err() {
         env::set_var("RUST_LOG", "debug,actix_web=debug");
    }
    env_logger::init();

    let args = Args::parse();

    ort::init()
        .with_execution_providers(
            [
                ort::TensorRTExecutionProvider::default().build(),
                ort::CUDAExecutionProvider::default().build(),
                ort::DirectMLExecutionProvider::default().build(),
            ]
        ).commit().expect("Failed to initialize ORT");

    let mut model_manager = SafeModelManager::new();
    model_registry::register_default_models(&mut model_manager).await;

    preload_models(&model_manager).await;

    let state = web::Data::new(AppState {
        model_manager
    });

    info!("Starting server on port {}...", args.port);

    let state_for_idle = state.clone();
    let model_ttl = 10 * 60;
    let poll_interval = 10;
    tokio::spawn(async move {
        idle_shutdown_task(state_for_idle, model_ttl, poll_interval).await
    });

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(web::JsonConfig::default().error_handler(|err, _req| {
                error!("JSON 解析错误: {:?}", err);
                actix_web::error::InternalError::from_response(
                    "",
                    HttpResponse::BadRequest().json(json!({
                        "error": "请求格式错误",
                        "details": err.to_string()
                    }))
                ).into()
            }))
            .app_data(web::FormConfig::default().error_handler(|err, _req| {
                error!("Form 解析错误: {:?}", err);
                actix_web::error::InternalError::from_response(
                    "",
                    HttpResponse::BadRequest().json(json!({
                        "error": "表单解析错误",
                        "details": err.to_string()
                    }))
                ).into()
            }))
            .app_data(state.clone())
            .route("/", web::get().to(|| async { 
                HttpResponse::Ok().json(json!({"message": "Immich ML"}))
            }))
            .route("/ping", web::get().to(|| async {
                HttpResponse::Ok().body("pong")
            }))
            .route("/predict", web::post().to(predict))
    })
    .bind(("0.0.0.0", args.port))?
    .run()
    .await
}