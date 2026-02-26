use crate::state::pool_store;
use crate::worker::InferRequest;
use redis_module::{Context, RedisError, RedisResult, RedisString, RedisValue};

/// INFER.GENERATE <token_key> [max_tokens] [temperature]
///
/// Read pre-tokenized uint32 data from a Redis STRING key,
/// run generation on a worker thread, and return the result.
/// Redis remains responsive during inference.
pub fn infer_generate(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
    if args.len() < 2 || args.len() > 4 {
        return Err(RedisError::WrongArity);
    }

    // Check pool exists (model loaded)
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

    // Copy key name to owned String (args may be freed after command returns)
    let token_key_name = args[1]
        .try_as_str()
        .map_err(|_| RedisError::Str("ERR invalid key name"))?
        .to_owned();

    // Block the client -- Redis returns immediately, client waits for reply
    let blocked_client = ctx.block_client();
    let thread_ctx =
        redis_module::ThreadSafeContext::with_blocked_client(blocked_client);

    let request = InferRequest {
        thread_ctx,
        token_key_name,
        max_tokens,
        temperature,
    };

    match pool.submit(request) {
        Ok(()) => Ok(RedisValue::NoReply),
        Err(e) => Err(RedisError::String(format!("ERR {e}"))),
    }
}
