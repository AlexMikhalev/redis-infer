use crate::InferError;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::path::Path;
use std::sync::Arc;

pub struct InferModel {
    pub backend: Arc<LlamaBackend>,
    pub model: Arc<LlamaModel>,
    pub model_path: String,
    pub n_vocab: i32,
}

impl InferModel {
    pub fn load(path: &str) -> Result<Self, InferError> {
        if !Path::new(path).exists() {
            return Err(InferError::ModelLoad {
                path: path.to_string(),
                reason: "file not found".to_string(),
            });
        }

        let backend = LlamaBackend::init().map_err(|e| InferError::ModelLoad {
            path: path.to_string(),
            reason: format!("{e:?}"),
        })?;

        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, path, &params).map_err(|e| {
            InferError::ModelLoad {
                path: path.to_string(),
                reason: format!("{e:?}"),
            }
        })?;

        let n_vocab = model.n_vocab();

        Ok(Self {
            backend: Arc::new(backend),
            model: Arc::new(model),
            model_path: path.to_string(),
            n_vocab,
        })
    }
}
