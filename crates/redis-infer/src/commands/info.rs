use redis_module::{Context, RedisResult, RedisString};

pub fn infer_info(_ctx: &Context, _args: Vec<RedisString>) -> RedisResult {
    Ok("redis-infer v0.1.0 (Rust)".into())
}
