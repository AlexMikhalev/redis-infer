use infer_engine::llama::generate::{generate_with_context, GenerateParams};
use infer_engine::llama::model::InferModel;
use infer_engine::tokens::bytes_to_tokens;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::model::AddBos;
use redis_module::{RedisError, RedisValue, ThreadSafeContext};
use std::num::NonZeroU32;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

/// Source of token data for inference.
pub enum TokenSource {
    /// Pre-tokenized: read packed uint32 from this Redis key.
    PreTokenized(String),
    /// Raw text: read UTF-8 string from this Redis key, tokenize at runtime.
    RawText(String),
}

/// Intermediate result from reading Redis key under GIL.
enum EitherTokens {
    Tokens(Vec<llama_cpp_2::token::LlamaToken>),
    Text(String),
}

/// A request to generate text.
pub struct InferRequest {
    pub thread_ctx: ThreadSafeContext<redis_module::BlockedClient>,
    pub source: TokenSource,
    pub max_tokens: i32,
    pub temperature: f32,
}

/// A bounded thread pool where each worker owns a pre-created LlamaContext.
pub struct WorkerPool {
    sender: Option<mpsc::SyncSender<InferRequest>>,
    handles: Vec<thread::JoinHandle<()>>,
    pool_size: usize,
}

impl WorkerPool {
    pub fn new(pool_size: usize, model: Arc<InferModel>, context_size: u32) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<InferRequest>(pool_size * 2);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut handles = Vec::with_capacity(pool_size);

        for worker_id in 0..pool_size {
            let recv = Arc::clone(&receiver);
            let model = Arc::clone(&model);

            let handle = thread::Builder::new()
                .name(format!("infer-worker-{}", worker_id))
                .spawn(move || {
                    // Each worker creates its own LlamaContext (not Send, lives on this thread)
                    let ctx_params = LlamaContextParams::default()
                        .with_n_ctx(NonZeroU32::new(context_size));

                    let mut llama_ctx =
                        match model.model.new_context(&model.backend, ctx_params) {
                            Ok(ctx) => ctx,
                            Err(e) => {
                                eprintln!(
                                    "redis-infer: worker-{} failed to create context: {:?}",
                                    worker_id, e
                                );
                                return;
                            }
                        };

                    loop {
                        let request = {
                            let lock = recv.lock().unwrap();
                            match lock.recv() {
                                Ok(req) => req,
                                Err(_) => break, // Channel closed, shut down
                            }
                        };

                        Self::handle_request(&model, &mut llama_ctx, request);
                    }
                })
                .expect("failed to spawn inference worker thread");

            handles.push(handle);
        }

        Self {
            sender: Some(sender),
            handles,
            pool_size,
        }
    }

    fn handle_request(
        model: &InferModel,
        llama_ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
        request: InferRequest,
    ) {
        // Step 1: Acquire GIL, read data from Redis, release GIL
        let tokens = {
            let locked_ctx = request.thread_ctx.lock();
            let key_name = match &request.source {
                TokenSource::PreTokenized(k) | TokenSource::RawText(k) => k.as_str(),
            };
            let rs_key = locked_ctx.create_string(key_name.as_bytes());
            let key = locked_ctx.open_key(&rs_key);
            let result = match key.read() {
                Ok(Some(data)) => {
                    match &request.source {
                        TokenSource::PreTokenized(_) => {
                            // Path A: data is packed uint32 tokens -- just parse
                            match bytes_to_tokens(data) {
                                Ok(tokens) => Ok(EitherTokens::Tokens(tokens)),
                                Err(e) => Err(format!("ERR {e}")),
                            }
                        }
                        TokenSource::RawText(_) => {
                            // Path B: data is UTF-8 text -- copy to owned String
                            match std::str::from_utf8(data) {
                                Ok(text) => Ok(EitherTokens::Text(text.to_owned())),
                                Err(e) => Err(format!("ERR invalid UTF-8: {e}")),
                            }
                        }
                    }
                }
                Ok(None) => Err("ERR key not found or empty".to_string()),
                Err(e) => Err(format!("ERR reading key: {e}")),
            };
            drop(key);
            drop(locked_ctx); // Release GIL
            result
        };
        // GIL is now released. Redis main thread is unblocked.

        // Resolve tokens (Path B needs runtime tokenization here, GIL-free)
        let tokens = match tokens {
            Err(err_msg) => {
                request
                    .thread_ctx
                    .reply(Err(RedisError::String(err_msg)));
                return;
            }
            Ok(either) => match either {
                EitherTokens::Tokens(t) => t,
                EitherTokens::Text(text) => {
                    // Runtime tokenization (Path B) -- happens on worker thread, no GIL
                    match model.model.str_to_token(&text, AddBos::Always) {
                        Ok(tokens) => tokens,
                        Err(e) => {
                            request.thread_ctx.reply(Err(RedisError::String(
                                format!("ERR tokenization failed: {e:?}"),
                            )));
                            return;
                        }
                    }
                }
            },
        };

        // Log what we're feeding to the model (proof: these are the actual token IDs)
        let source_label = match &request.source {
            TokenSource::PreTokenized(k) => format!("PreTokenized({})", k),
            TokenSource::RawText(k) => format!("RawText({})", k),
        };
        eprintln!(
            "redis-infer: inference source={} n_tokens={} first_8={:?}{}",
            source_label,
            tokens.len(),
            &tokens[..tokens.len().min(8)],
            if tokens.len() > 8 { "..." } else { "" }
        );

        // Step 2: Run inference (seconds, GIL-free)
        llama_ctx.clear_kv_cache();

        let params = GenerateParams {
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            ..Default::default()
        };

        let result = generate_with_context(llama_ctx, &model.model, &tokens, &params);

        // Step 3: Reply
        match result {
            Ok(text) => {
                request.thread_ctx.reply(Ok(RedisValue::BulkString(text)));
            }
            Err(e) => {
                request
                    .thread_ctx
                    .reply(Err(RedisError::String(format!("ERR {e}"))));
            }
        }
    }

    pub fn submit(&self, request: InferRequest) -> Result<(), String> {
        self.sender
            .as_ref()
            .ok_or_else(|| "worker pool is shutting down".to_string())?
            .try_send(request)
            .map_err(|_| "all inference workers are busy".to_string())
    }

    pub fn pool_size(&self) -> usize {
        self.pool_size
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        // Drop sender first -- closes channel so workers see Err on recv and exit
        self.sender.take();
        // Join all worker threads to ensure Metal/GPU resources are fully released
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}
