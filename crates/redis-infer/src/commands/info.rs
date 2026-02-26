use crate::state::model_store;
use redis_module::{Context, RedisResult, RedisString};

pub fn infer_info(_ctx: &Context, _args: Vec<RedisString>) -> RedisResult {
    let guard = model_store().read().map_err(|_| {
        redis_module::RedisError::Str("ERR model lock poisoned")
    })?;

    let info = match guard.as_ref() {
        Some(model) => format!(
            "redis-infer v0.1.0 (Rust)\nmodel: {}\nvocab: {}",
            model.model_path, model.n_vocab
        ),
        None => "redis-infer v0.1.0 (Rust)\nmodel: none".to_string(),
    };

    Ok(info.into())
}
