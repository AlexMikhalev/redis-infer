use crate::InferError;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::path::Path;
use std::sync::{Arc, OnceLock};

static BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();

fn get_backend() -> Result<Arc<LlamaBackend>, InferError> {
    if let Some(backend) = BACKEND.get() {
        return Ok(Arc::clone(backend));
    }
    let b = LlamaBackend::init().map_err(|e| InferError::ModelLoad {
        path: String::new(),
        reason: format!("backend init: {e:?}"),
    })?;
    let arc = Arc::new(b);
    // If another thread beat us, that's fine -- we just discard ours
    let _ = BACKEND.set(Arc::clone(&arc));
    Ok(BACKEND.get().map(Arc::clone).unwrap_or(arc))
}

pub struct InferModel {
    pub backend: Arc<LlamaBackend>,
    pub model: Arc<LlamaModel>,
    pub model_path: String,
    pub n_vocab: i32,
}

impl InferModel {
    /// Load a GGUF model. `gpu_layers` controls Metal/CUDA offload (None = default, Some(0) = CPU only).
    pub fn load(path: &str, gpu_layers: Option<u32>) -> Result<Self, InferError> {
        if !Path::new(path).exists() {
            return Err(InferError::ModelLoad {
                path: path.to_string(),
                reason: "file not found".to_string(),
            });
        }

        let backend = get_backend()?;

        let mut params = LlamaModelParams::default();
        if let Some(layers) = gpu_layers {
            params = params.with_n_gpu_layers(layers);
        }
        let model = LlamaModel::load_from_file(&backend, path, &params).map_err(|e| {
            InferError::ModelLoad {
                path: path.to_string(),
                reason: format!("{e:?}"),
            }
        })?;

        let n_vocab = model.n_vocab();

        Ok(Self {
            backend,
            model: Arc::new(model),
            model_path: path.to_string(),
            n_vocab,
        })
    }
}
