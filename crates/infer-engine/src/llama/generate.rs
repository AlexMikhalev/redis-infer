use crate::InferError;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;

pub struct GenerateParams {
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub context_size: u32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            context_size: 4096,
        }
    }
}

/// Run generation on the model, creating a temporary context.
/// This is the simple (blocking) path for Phase 2.
pub fn generate(
    model: &LlamaModel,
    backend: &LlamaBackend,
    input_tokens: &[LlamaToken],
    params: &GenerateParams,
) -> Result<String, InferError> {
    let ctx_params =
        LlamaContextParams::default().with_n_ctx(NonZeroU32::new(params.context_size));

    let mut ctx = model
        .new_context(backend, ctx_params)
        .map_err(|e| InferError::ContextCreate(format!("{e:?}")))?;

    generate_with_context(&mut ctx, model, input_tokens, params)
}

/// Run generation using an existing context (for thread pool reuse in Phase 3).
/// The caller is responsible for calling `ctx.clear_kv_cache()` before this if reusing.
pub fn generate_with_context(
    ctx: &mut LlamaContext<'_>,
    model: &LlamaModel,
    input_tokens: &[LlamaToken],
    params: &GenerateParams,
) -> Result<String, InferError> {
    if input_tokens.is_empty() {
        return Err(InferError::InvalidTokenData(
            "empty token sequence".to_string(),
        ));
    }

    let n_tokens = input_tokens.len();
    let mut batch = LlamaBatch::new(n_tokens.max(512), 1);

    // Add all input tokens to the batch
    let last_idx = n_tokens - 1;
    for (i, &token) in input_tokens.iter().enumerate() {
        batch
            .add(token, i as i32, &[0], i == last_idx)
            .map_err(|e| InferError::Decode(format!("batch add: {e:?}")))?;
    }

    // Prefill: decode all input tokens
    ctx.decode(&mut batch)
        .map_err(|e| InferError::Decode(format!("prefill: {e:?}")))?;

    // Set up sampler
    let mut sampler = if params.temperature <= 0.0 {
        LlamaSampler::greedy()
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::top_k(params.top_k),
            LlamaSampler::top_p(params.top_p, 1),
            LlamaSampler::temp(params.temperature),
            LlamaSampler::dist(42),
        ])
    };

    // Generation loop
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = batch.n_tokens();

    for _ in 0..params.max_tokens {
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        match model.token_to_piece(token, &mut decoder, true, None) {
            Ok(piece) => output.push_str(&piece),
            Err(_) => {} // skip unprintable tokens
        }

        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| InferError::Decode(format!("gen batch add: {e:?}")))?;
        n_cur += 1;

        ctx.decode(&mut batch)
            .map_err(|e| InferError::Decode(format!("gen decode: {e:?}")))?;
    }

    Ok(output)
}
