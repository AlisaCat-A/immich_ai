use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use chinese_clip_rs::{ImageProcessor, TextProcessor};
use log::info;
use tokio::sync::Mutex;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

// Import new models
pub mod face_detection;
pub mod face_recognition;
pub mod multilingual_clip;

use face_detection::FaceDetectionModel;
use face_recognition::FaceRecognitionModel;
use multilingual_clip::SiglipModel;

// 模型特征
pub trait Model: Send + Sync {
    fn process(&self, input: &ModelInput) -> Result<ModelOutput>;
}

// 模型输入枚举
#[derive(Debug)]
pub enum ModelInput {
    Image(Vec<u8>),
    Text(String),
}

// 模型输出
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ModelOutput {
    Clip(Vec<f32>),
    FaceDetection(face_detection::FaceDetectionOutput),
    FaceRecognition(face_recognition::FaceRecognitionOutput),
}

// 图像处理模型 (Chinese CLIP)
pub struct ImageModel {
    processor: ImageProcessor,
    image_size: (u32, u32),
}

impl ImageModel {
    pub fn new(model_path: &str, image_size: (u32, u32)) -> Result<Self> {
        Ok(Self {
            processor: ImageProcessor::new(&model_path, image_size)
                .context("Failed to create ImageProcessor")?,
            image_size,
        })
    }
}

impl Model for ImageModel {
    fn process(&self, input: &ModelInput) -> Result<ModelOutput> {
        match input {
            ModelInput::Image(image_data) => {
                let features = self.processor.process_image(image_data)?;
                Ok(ModelOutput::Clip(features))
            }
            _ => Err(anyhow::anyhow!("Invalid input type for ImageModel")),
        }
    }
}

// 文本处理模型 (Chinese CLIP)
pub struct TextModel {
    processor: TextProcessor,
    max_length: usize,
}

impl TextModel {
    pub fn new(model_path: &str, tokenizer_path: &str, max_length: usize) -> Result<Self> {
        Ok(Self {
            processor: TextProcessor::new(&model_path, &tokenizer_path, max_length)
                .context("Failed to create TextProcessor")?,
            max_length,
        })
    }
}

impl Model for TextModel {
    fn process(&self, input: &ModelInput) -> Result<ModelOutput> {
        match input {
            ModelInput::Text(text) => {
                let features = self.processor.process_text(text)?;
                Ok(ModelOutput::Clip(features))
            }
            _ => Err(anyhow::anyhow!("Invalid input type for TextModel")),
        }
    }
}

// Wrapper for Face Detection
pub struct FaceDetectionModelWrapper {
    inner: FaceDetectionModel,
}

impl FaceDetectionModelWrapper {
    pub fn new(model_path: &str) -> Result<Self> {
        Ok(Self {
            inner: FaceDetectionModel::new(model_path)?,
        })
    }
}

impl Model for FaceDetectionModelWrapper {
    fn process(&self, input: &ModelInput) -> Result<ModelOutput> {
        match input {
            ModelInput::Image(image_data) => {
                let output = self.inner.process(image_data)?;
                Ok(ModelOutput::FaceDetection(output))
            }
            _ => Err(anyhow::anyhow!("Invalid input for Face Detection")),
        }
    }
}

// Wrapper for Face Recognition
pub struct FaceRecognitionModelWrapper {
    inner: FaceRecognitionModel,
}

impl FaceRecognitionModelWrapper {
    pub fn new(model_path: &str) -> Result<Self> {
        Ok(Self {
            inner: FaceRecognitionModel::new(model_path)?,
        })
    }
}

impl Model for FaceRecognitionModelWrapper {
    fn process(&self, input: &ModelInput) -> Result<ModelOutput> {
        match input {
            ModelInput::Image(image_data) => {
                let img = image::load_from_memory(image_data)?;
                let output = self.inner.process(&img)?;
                Ok(ModelOutput::FaceRecognition(output))
            }
            _ => Err(anyhow::anyhow!("Invalid input for Face Recognition")),
        }
    }
}

// Wrapper for SigLIP (Multilingual CLIP)
pub struct SiglipModelWrapper {
    inner: SiglipModel,
    is_text: bool,
}

impl SiglipModelWrapper {
    pub fn new_vision(model_path: &str) -> Result<Self> {
        Ok(Self {
            inner: SiglipModel::new_vision(model_path)?,
            is_text: false,
        })
    }

    pub fn new_text(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        Ok(Self {
            inner: SiglipModel::new_text(model_path, tokenizer_path)?,
            is_text: true,
        })
    }
}

impl Model for SiglipModelWrapper {
    fn process(&self, input: &ModelInput) -> Result<ModelOutput> {
        if self.is_text {
             match input {
                ModelInput::Text(text) => {
                    let features = self.inner.process_text(text)?;
                    Ok(ModelOutput::Clip(features))
                }
                _ => Err(anyhow::anyhow!("Invalid input type for Siglip Text")),
            }
        } else {
             match input {
                ModelInput::Image(image_data) => {
                    let features = self.inner.process_image(image_data)?;
                    Ok(ModelOutput::Clip(features))
                }
                _ => Err(anyhow::anyhow!("Invalid input type for Siglip Vision")),
            }
        }
    }
}


// 模型实例包装
struct ModelInstance {
    model: Arc<dyn Model>,
    last_used: Instant,
}

// 模型管理器
pub struct ModelManager {
    models: HashMap<String, ModelInstance>,
    model_configs: HashMap<String, ModelConfig>,
}

#[derive(Clone)]
pub struct ModelConfig {
    pub model_type: ModelType,
    pub model_path: String,
    pub tokenizer_path: Option<String>,
    pub image_size: Option<(u32, u32)>,
    pub max_length: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelType {
    Image,
    Text,
    FaceDetection,
    FaceRecognition,
    SiglipVision,
    SiglipText,
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            model_configs: HashMap::new(),
        }
    }

    pub fn register_model(&mut self, name: String, config: ModelConfig) {
        self.model_configs.insert(name, config);
    }

    // Allow directly injecting a model instance (useful for testing or manual loading)
    pub fn register_model_instance(&mut self, name: String, model: Arc<dyn Model>) {
        self.models.insert(name, ModelInstance {
            model,
            last_used: Instant::now(),
        });
    }

    pub async fn get_or_load_model(&mut self, name: &str) -> Result<Arc<dyn Model>> {
        if let Some(instance) = self.models.get_mut(name) {
            instance.last_used = Instant::now();
            return Ok(Arc::clone(&instance.model));
        }

        let config = self.model_configs.get(name)
            .ok_or_else(|| anyhow::anyhow!("Model config not found: {}", name))?;

        let model = self.load_model(config).await?;

        self.models.insert(name.to_string(), ModelInstance {
            model: Arc::clone(&model),
            last_used: Instant::now(),
        });

        Ok(model)
    }

    async fn load_model(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        info!("Loading model type: {:?}", config.model_type);
        match config.model_type {
            ModelType::Image => {
                let image_size = config.image_size.ok_or_else(|| 
                    anyhow::anyhow!("Image size not specified"))?;
                Ok(Arc::new(ImageModel::new(
                    &config.model_path,
                    image_size,
                )?))
            }
            ModelType::Text => {
                let tokenizer_path = config.tokenizer_path.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Tokenizer path not specified"))?;
                let max_length = config.max_length
                    .ok_or_else(|| anyhow::anyhow!("Max length not specified"))?;
                Ok(Arc::new(TextModel::new(
                    &config.model_path,
                    &tokenizer_path,
                    max_length,
                )?))
            }
            ModelType::FaceDetection => {
                Ok(Arc::new(FaceDetectionModelWrapper::new(&config.model_path)?))
            }
            ModelType::FaceRecognition => {
                Ok(Arc::new(FaceRecognitionModelWrapper::new(&config.model_path)?))
            }
            ModelType::SiglipVision => {
                Ok(Arc::new(SiglipModelWrapper::new_vision(&config.model_path)?))
            }
            ModelType::SiglipText => {
                 let tokenizer_path = config.tokenizer_path.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Tokenizer path not specified"))?;
                Ok(Arc::new(SiglipModelWrapper::new_text(&config.model_path, tokenizer_path)?))
            }
        }
    }

    pub async fn cleanup_expired_models(&mut self, ttl: Duration) {
        let now = Instant::now();
        self.models.retain(|name, instance| {
            let retain = now.duration_since(instance.last_used) < ttl;
            if !retain {
                info!("清理过期模型: {}", name);
            }
            retain
        });
    }
}

// 线程安全的模型管理器包装
pub struct SafeModelManager {
    inner: Arc<Mutex<ModelManager>>,
}

impl SafeModelManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ModelManager::new())),
        }
    }

    pub async fn register_model(&self, name: String, config: ModelConfig) {
        let mut manager = self.inner.lock().await;
        manager.register_model(name, config);
    }

    pub async fn register_model_instance(&self, name: String, model: Arc<dyn Model>) {
        let mut manager = self.inner.lock().await;
        manager.register_model_instance(name, model);
    }

    pub async fn get_model(&self, name: &str) -> Result<Arc<dyn Model>> {
        let mut manager = self.inner.lock().await;
        manager.get_or_load_model(name).await
    }

    pub async fn cleanup_expired_models(&self, ttl: Duration) {
        let mut manager = self.inner.lock().await;
        manager.cleanup_expired_models(ttl).await;
    }
}
