use crate::state::{model_store, pool_store};
use crate::worker::WorkerPool;
use infer_engine::llama::model::InferModel;
use redis_module::{Context, RedisError, RedisResult, RedisString};
use std::sync::Arc;

/// INFER.LOAD <model_path> [num_workers] [context_size]
///
/// Load a GGUF model from the filesystem and create a worker pool.
/// Blocks Redis during loading (expected 2-10 seconds).
/// This is an admin operation run once.
pub fn infer_load(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 || args.len() > 4 {
        return Err(RedisError::WrongArity);
    }
    let path = args[1]
        .try_as_str()
        .map_err(|_| RedisError::Str("ERR invalid model path (not valid UTF-8)"))?;

    let num_workers = if args.len() >= 3 {
        args[2]
            .try_as_str()
            .map_err(|_| RedisError::Str("ERR invalid num_workers"))?
            .parse::<usize>()
            .map_err(|_| RedisError::Str("ERR num_workers must be a positive integer"))?
    } else {
        num_cpus::get()
    };

    let context_size = if args.len() >= 4 {
        args[3]
            .try_as_str()
            .map_err(|_| RedisError::Str("ERR invalid context_size"))?
            .parse::<u32>()
            .map_err(|_| RedisError::Str("ERR context_size must be a positive integer"))?
    } else {
        4096
    };

    ctx.log_notice(&format!("redis-infer: loading model from {}", path));

    let model = InferModel::load(path)
        .map_err(|e| RedisError::String(format!("ERR loading model: {e}")))?;

    let n_vocab = model.n_vocab;
    let model_path = model.model_path.clone();
    let model = Arc::new(model);

    // Store model
    {
        let mut guard = model_store()
            .write()
            .map_err(|_| RedisError::Str("ERR model lock poisoned"))?;
        *guard = Some(Arc::clone(&model));
    }

    // Create worker pool
    ctx.log_notice(&format!(
        "redis-infer: creating {} worker threads (context_size={})",
        num_workers, context_size
    ));

    let pool = WorkerPool::new(num_workers, model, context_size);

    {
        let mut guard = pool_store()
            .write()
            .map_err(|_| RedisError::Str("ERR pool lock poisoned"))?;
        *guard = Some(Arc::new(pool));
    }

    ctx.log_notice(&format!(
        "redis-infer: model loaded (vocab={}, workers={}, ctx={}, path={})",
        n_vocab, num_workers, context_size, model_path
    ));

    Ok("OK".into())
}
