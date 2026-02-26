use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferError {
    #[error("failed to load model from '{path}': {reason}")]
    ModelLoad { path: String, reason: String },

    #[error("failed to create inference context: {0}")]
    ContextCreate(String),

    #[error("decode failed: {0}")]
    Decode(String),

    #[error("model not loaded")]
    NoModel,

    #[error("all inference contexts are busy")]
    PoolExhausted,

    #[error("invalid token data: {0}")]
    InvalidTokenData(String),
}
