use serde_json::Error;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("I/O failure")]
    Io(#[from] std::io::Error),
    #[error("Deserialization failure")]
    Deserialization(#[from] Error),
    #[error("Vocabulary version mismatch")]
    Vocabulary(String),
}
