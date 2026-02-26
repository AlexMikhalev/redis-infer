use infer_engine::llama::model::InferModel;
use std::sync::{Arc, OnceLock, RwLock};

static MODEL: OnceLock<RwLock<Option<Arc<InferModel>>>> = OnceLock::new();

pub fn model_store() -> &'static RwLock<Option<Arc<InferModel>>> {
    MODEL.get_or_init(|| RwLock::new(None))
}
