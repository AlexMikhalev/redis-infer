use crate::state::{model_store, pool_store};
use redis_module::{Context, RedisError, RedisResult, RedisString};

pub fn infer_info(_ctx: &Context, _args: Vec<RedisString>) -> RedisResult {
    let model_guard = model_store()
        .read()
        .map_err(|_| RedisError::Str("ERR model lock poisoned"))?;
    let pool_guard = pool_store()
        .read()
        .map_err(|_| RedisError::Str("ERR pool lock poisoned"))?;

    let info = match (model_guard.as_ref(), pool_guard.as_ref()) {
        (Some(model), Some(pool)) => format!(
            "redis-infer v0.1.0 (Rust)\nmodel: {}\nvocab: {}\nworkers: {}",
            model.model_path,
            model.n_vocab,
            pool.pool_size()
        ),
        (Some(model), None) => format!(
            "redis-infer v0.1.0 (Rust)\nmodel: {}\nvocab: {}\nworkers: 0",
            model.model_path, model.n_vocab
        ),
        _ => "redis-infer v0.1.0 (Rust)\nmodel: none".to_string(),
    };

    Ok(info.into())
}
