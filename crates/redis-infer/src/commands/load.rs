use crate::state::model_store;
use infer_engine::llama::model::InferModel;
use redis_module::{Context, RedisError, RedisResult, RedisString};
use std::sync::Arc;

/// INFER.LOAD <model_path>
///
/// Load a GGUF model from the filesystem. Blocks Redis during loading
/// (expected 2-5 seconds). This is an admin operation run once.
pub fn infer_load(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() != 2 {
        return Err(RedisError::WrongArity);
    }
    let path = args[1]
        .try_as_str()
        .map_err(|_| RedisError::Str("ERR invalid model path (not valid UTF-8)"))?;

    ctx.log_notice(&format!("redis-infer: loading model from {}", path));

    let model = InferModel::load(path)
        .map_err(|e| RedisError::String(format!("ERR loading model: {e}")))?;

    let n_vocab = model.n_vocab;
    let model_path = model.model_path.clone();

    let mut guard = model_store()
        .write()
        .map_err(|_| RedisError::Str("ERR model lock poisoned"))?;
    *guard = Some(Arc::new(model));

    ctx.log_notice(&format!(
        "redis-infer: model loaded (vocab={}, path={})",
        n_vocab, model_path
    ));

    Ok("OK".into())
}
