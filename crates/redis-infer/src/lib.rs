use redis_module::{redis_module, Context, RedisString, Status};

mod commands;
mod state;
mod worker;

fn init(_ctx: &Context, _args: &[RedisString]) -> Status {
    Status::Ok
}

fn deinit(ctx: &Context) -> Status {
    ctx.log_notice("redis-infer: shutting down, draining worker pool...");

    // Drop the pool (closes channel, workers exit)
    if let Ok(mut guard) = state::pool_store().write() {
        *guard = None;
    }

    // Drop the model
    if let Ok(mut guard) = state::model_store().write() {
        *guard = None;
    }

    ctx.log_notice("redis-infer: shutdown complete");
    Status::Ok
}

redis_module! {
    name: "redis-infer",
    version: 1,
    allocator: (redis_module::alloc::RedisAlloc, redis_module::alloc::RedisAlloc),
    data_types: [],
    init: init,
    deinit: deinit,
    commands: [
        ["infer.info", commands::info::infer_info, "readonly fast", 0, 0, 0],
        ["infer.load", commands::load::infer_load, "write deny-oom", 0, 0, 0],
        ["infer.generate", commands::generate::infer_generate, "write", 1, 1, 1],
        ["infer.generate_text", commands::generate_text::infer_generate_text, "write", 1, 1, 1],
    ],
}
