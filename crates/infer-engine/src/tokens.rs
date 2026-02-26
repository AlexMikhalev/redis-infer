use crate::InferError;
use llama_cpp_2::token::LlamaToken;

/// Convert raw bytes (packed little-endian uint32) to LlamaToken vec.
/// Pre-tokenized data is stored in Redis as binary STRING keys.
pub fn bytes_to_tokens(data: &[u8]) -> Result<Vec<LlamaToken>, InferError> {
    if data.len() % 4 != 0 {
        return Err(InferError::InvalidTokenData(format!(
            "token data length {} is not a multiple of 4",
            data.len()
        )));
    }
    let tokens: Vec<LlamaToken> = data
        .chunks_exact(4)
        .map(|chunk| {
            let id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            LlamaToken::new(id as i32)
        })
        .collect();
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_tokens_valid() {
        // Token IDs 1, 2, 3 as little-endian uint32
        let data = vec![
            1, 0, 0, 0, // 1
            2, 0, 0, 0, // 2
            3, 0, 0, 0, // 3
        ];
        let tokens = bytes_to_tokens(&data).unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], LlamaToken::new(1));
        assert_eq!(tokens[1], LlamaToken::new(2));
        assert_eq!(tokens[2], LlamaToken::new(3));
    }

    #[test]
    fn test_bytes_to_tokens_empty() {
        let tokens = bytes_to_tokens(&[]).unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_bytes_to_tokens_invalid_length() {
        let data = vec![1, 2, 3]; // 3 bytes, not a multiple of 4
        assert!(bytes_to_tokens(&data).is_err());
    }

    #[test]
    fn test_bytes_to_tokens_large_id() {
        // Token ID 151643 (a real Qwen token ID) as little-endian uint32
        let id: u32 = 151643;
        let bytes = id.to_le_bytes();
        let tokens = bytes_to_tokens(&bytes).unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], LlamaToken::new(151643));
    }
}
