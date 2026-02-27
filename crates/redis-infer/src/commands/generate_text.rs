use crate::state::pool_store;
use crate::worker::{InferRequest, TokenSource};
use redis_module::{Context, RedisError, RedisResult, RedisString, RedisValue};

/// INFER.GENERATE_TEXT <text_key> [max_tokens] [temperature]
///
/// Read raw UTF-8 text from a Redis STRING key,
/// tokenize it at runtime on the worker thread, then run generation.
/// This is the "Path B" baseline for A/B testing against pre-tokenized INFER.GENERATE.
pub fn infer_generate_text(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 || args.len() > 4 {
        return Err(RedisError::WrongArity);
    }

    let guard = pool_store()
        .read()
        .map_err(|_| RedisError::Str("ERR pool lock poisoned"))?;
    let pool = guard
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

    let text_key_name = args[1]
        .try_as_str()
        .map_err(|_| RedisError::Str("ERR invalid key name"))?
        .to_owned();

    let blocked_client = ctx.block_client();
    let thread_ctx =
        redis_module::ThreadSafeContext::with_blocked_client(blocked_client);

    let request = InferRequest {
        thread_ctx,
        source: TokenSource::RawText(text_key_name),
        max_tokens,
        temperature,
    };

    match pool.submit(request) {
        Ok(()) => Ok(RedisValue::NoReply),
        Err(e) => Err(RedisError::String(format!("ERR {e}"))),
    }
}
