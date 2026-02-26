use crate::state::{model_store, pool_store};
use crate::worker::WorkerPool;
use infer_engine::llama::model::InferModel;
use redis_module::{Context, RedisError, RedisResult, RedisString};
use std::sync::Arc;

/// INFER.LOAD <model_path> [num_workers] [context_size] [gpu_layers]
///
/// Load a GGUF model from the filesystem and create a worker pool.
/// Blocks Redis during loading (expected 2-10 seconds).
/// This is an admin operation run once.
///
/// gpu_layers: number of layers to offload to GPU (Metal/CUDA).
///   Default uses llama.cpp default. Set to 0 for CPU-only.
pub fn infer_load(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 || args.len() > 5 {
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

    let gpu_layers = if args.len() >= 5 {
        Some(
            args[4]
                .try_as_str()
                .map_err(|_| RedisError::Str("ERR invalid gpu_layers"))?
                .parse::<u32>()
                .map_err(|_| RedisError::Str("ERR gpu_layers must be a non-negative integer"))?,
        )
    } else {
        None
    };

    // Drain existing pool and model before loading new one.
    // This ensures Metal/GPU resources from old LlamaContexts are fully released.
    {
        let mut guard = pool_store()
            .write()
            .map_err(|_| RedisError::Str("ERR pool lock poisoned"))?;
        if guard.is_some() {
            ctx.log_notice("redis-infer: draining existing worker pool...");
        }
        *guard = None; // Drop pool -> closes channel -> workers exit
    }
    {
        let mut guard = model_store()
            .write()
            .map_err(|_| RedisError::Str("ERR model lock poisoned"))?;
        *guard = None; // Drop old model
    }

    ctx.log_notice(&format!("redis-infer: loading model from {}", path));

    let model = InferModel::load(path, gpu_layers)
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
