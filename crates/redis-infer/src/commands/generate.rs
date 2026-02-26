use crate::state::model_store;
use crate::tokens::bytes_to_tokens;
use infer_engine::llama::generate::{generate, GenerateParams};
use redis_module::{Context, RedisError, RedisResult, RedisString};

/// INFER.GENERATE <token_key> [max_tokens] [temperature]
///
/// Phase 2: blocking single-threaded generation.
/// Reads pre-tokenized uint32 data from a Redis STRING key,
/// runs generation, and returns the result.
///
/// WARNING: This blocks Redis during inference (10-50 seconds on CPU).
/// Phase 3 will add non-blocking threaded inference.
pub fn infer_generate(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 || args.len() > 4 {
        return Err(RedisError::WrongArity);
    }

    // Check model is loaded
    let guard = model_store()
        .read()
        .map_err(|_| RedisError::Str("ERR model lock poisoned"))?;
    let model = guard
        .as_ref()
        .ok_or(RedisError::Str("ERR no model loaded, use INFER.LOAD first"))?;

    let max_tokens = if args.len() >= 3 {
        args[2]
            .try_as_str()
            .map_err(|_| RedisError::Str("ERR invalid max_tokens"))?
            .parse::<i32>()
            .map_err(|_| RedisError::Str("ERR max_tokens must be an integer"))?
    } else {
        256
    };

    let temperature = if args.len() >= 4 {
        args[3]
            .try_as_str()
            .map_err(|_| RedisError::Str("ERR invalid temperature"))?
            .parse::<f32>()
            .map_err(|_| RedisError::Str("ERR temperature must be a float"))?
    } else {
        0.7
    };

    // Read token data from Redis key
    let key = ctx.open_key(&args[1]);
    let data = key
        .read()
        .map_err(|e| RedisError::String(format!("ERR reading key: {e}")))?
        .ok_or(RedisError::Str("ERR key not found or empty"))?;

    let tokens = bytes_to_tokens(data)
        .map_err(|e| RedisError::String(format!("ERR {e}")))?;

    ctx.log_notice(&format!(
        "redis-infer: generating from {} tokens, max_tokens={}, temp={}",
        tokens.len(),
        max_tokens,
        temperature
    ));

    let params = GenerateParams {
        max_tokens,
        temperature,
        ..Default::default()
    };

    let result = generate(&model.model, &model.backend, &tokens, &params)
        .map_err(|e| RedisError::String(format!("ERR inference: {e}")))?;

    Ok(result.into())
}
