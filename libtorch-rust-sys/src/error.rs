use std::fmt;

#[derive(Debug, Clone)]
pub enum TchError {
    /// Tensor operation error
    TensorError(String),
    /// Shape mismatch error
    ShapeError(String),
    /// Type mismatch error
    TypeError(String),
    /// Device error (e.g., CUDA not available)
    DeviceError(String),
    /// Index out of bounds
    IndexError(String),
    /// File I/O error
    FileError(String),
    /// Serialization/deserialization error
    SerializationError(String),
    /// Other errors
    Other(String),
}

impl fmt::Display for TchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TchError::TensorError(msg) => write!(f, "Tensor error: {}", msg),
            TchError::ShapeError(msg) => write!(f, "Shape error: {}", msg),
            TchError::TypeError(msg) => write!(f, "Type error: {}", msg),
            TchError::DeviceError(msg) => write!(f, "Device error: {}", msg),
            TchError::IndexError(msg) => write!(f, "Index error: {}", msg),
            TchError::FileError(msg) => write!(f, "File error: {}", msg),
            TchError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            TchError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for TchError {}

pub type Result<T> = std::result::Result<T, TchError>;
