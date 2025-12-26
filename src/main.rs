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

#[cfg(test)]
mod tests;

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

// Helper to crop face based on box
fn crop_face(image: &DynamicImage, bbox: &[f32; 4]) -> DynamicImage {
    let (x1, y1, x2, y2) = (bbox[0], bbox[1], bbox[2], bbox[3]);
    let width = (x2 - x1).max(1.0) as u32;
    let height = (y2 - y1).max(1.0) as u32;

    // Check bounds
    let (img_w, img_h) = image.dimensions();
    let x = x1.max(0.0) as u32;
    let y = y1.max(0.0) as u32;
    let w = width.min(img_w - x);
    let h = height.min(img_h - y);

    image.view(x, y, w, h).to_image().into()
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

                let output = model.process(&models::ModelInput::Image(image_data.data.to_vec()))
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

                let output = model.process(&models::ModelInput::Text(text.to_string()))
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
                let detection_model_name = &detection_entry.model_name;
                let det_model = state.model_manager.get_model(detection_model_name).await
                    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                let det_output = det_model.process(&models::ModelInput::Image(image_data.data.to_vec()))
                    .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                let faces_data = if let models::ModelOutput::FaceDetection(output) = det_output {
                    // Load original image for cropping
                    let original_img = image::load_from_memory(&image_data.data)
                        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                    if let Some(recognition_entry) = face_task.get(&ModelTypeEnum::Recognition) {
                        let rec_model_name = format!("{}_recognition", recognition_entry.model_name);
                         let rec_model = state.model_manager.get_model(&rec_model_name).await
                            .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

                        let mut detected_faces = Vec::new();

                        // Iterate over detected boxes
                        for (i, bbox) in output.boxes.iter().enumerate() {
                            let score = output.scores.get(i).cloned().unwrap_or(0.0);

                            // Crop face
                            // Ideally use landmarks for alignment, but basic crop is fallback
                            // TODO: Add full 5-point affine alignment here using output.landmarks[i]
                            let face_crop = crop_face(&original_img, bbox);

                            // Encode crop back to bytes or passed as image object?
                            // Our ModelInput expects Bytes or Text.
                            // We need to update ModelInput to accept DynamicImage or Bytes.
                            // Or re-encode to bytes (slow).
                            // Let's optimize: Update FaceRecognitionModelWrapper to accept ImageInput that can be bytes OR raw.

                            // For now, re-encode to memory is simplest (though slower).
                            let mut buffer = std::io::Cursor::new(Vec::new());
                            face_crop.write_to(&mut buffer, image::ImageOutputFormat::Jpeg(90))
                                .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
                            let crop_bytes = buffer.into_inner();

                            let rec_output = rec_model.process(&models::ModelInput::Image(crop_bytes))
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