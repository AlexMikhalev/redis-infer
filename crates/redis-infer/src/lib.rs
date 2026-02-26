use redis_module::{redis_module, Context, RedisString, Status};

mod commands;
mod state;
mod tokens;

fn init(_ctx: &Context, _args: &[RedisString]) -> Status {
    Status::Ok
}

fn deinit(_ctx: &Context) -> Status {
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
    ],
}
