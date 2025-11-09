use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, TchError};
use std::sync::Arc;

/// Internal storage for tensor data
pub struct Storage {
    data: Arc<StorageData>,
}

enum StorageData {
    Uint8(Vec<u8>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Bool(Vec<bool>),
}

impl Storage {
    /// Create new storage with the given size and dtype
    pub fn new(size: usize, dtype: DType, device: Device) -> Result<Self> {
        if !device.is_cpu() {
            return Err(TchError::DeviceError(
                "Only CPU device is currently supported".to_string(),
            ));
        }

        let data = match dtype {
            DType::Uint8 => StorageData::Uint8(vec![0u8; size]),
            DType::Int8 => StorageData::Int8(vec![0i8; size]),
            DType::Int16 => StorageData::Int16(vec![0i16; size]),
            DType::Int => StorageData::Int32(vec![0i32; size]),
            DType::Int64 => StorageData::Int64(vec![0i64; size]),
            DType::Float => StorageData::Float32(vec![0f32; size]),
            DType::Double => StorageData::Float64(vec![0f64; size]),
            DType::Bool => StorageData::Bool(vec![false; size]),
            _ => {
                return Err(TchError::TypeError(format!(
                    "Unsupported dtype: {:?}",
                    dtype
                )))
            }
        };

        Ok(Storage {
            data: Arc::new(data),
        })
    }

    /// Create storage from a vector of f32 values
    pub fn from_vec_f32(data: Vec<f32>) -> Self {
        Storage {
            data: Arc::new(StorageData::Float32(data)),
        }
    }

    /// Create storage from a vector of f64 values
    pub fn from_vec_f64(data: Vec<f64>) -> Self {
        Storage {
            data: Arc::new(StorageData::Float64(data)),
        }
    }

    /// Create storage from a vector of i32 values
    pub fn from_vec_i32(data: Vec<i32>) -> Self {
        Storage {
            data: Arc::new(StorageData::Int32(data)),
        }
    }

    /// Create storage from a vector of i64 values
    pub fn from_vec_i64(data: Vec<i64>) -> Self {
        Storage {
            data: Arc::new(StorageData::Int64(data)),
        }
    }

    /// Get the number of elements in storage
    pub fn size(&self) -> usize {
        match &*self.data {
            StorageData::Uint8(v) => v.len(),
            StorageData::Int8(v) => v.len(),
            StorageData::Int16(v) => v.len(),
            StorageData::Int32(v) => v.len(),
            StorageData::Int64(v) => v.len(),
            StorageData::Float32(v) => v.len(),
            StorageData::Float64(v) => v.len(),
            StorageData::Bool(v) => v.len(),
        }
    }

    /// Get the dtype of the storage
    pub fn dtype(&self) -> DType {
        match &*self.data {
            StorageData::Uint8(_) => DType::Uint8,
            StorageData::Int8(_) => DType::Int8,
            StorageData::Int16(_) => DType::Int16,
            StorageData::Int32(_) => DType::Int,
            StorageData::Int64(_) => DType::Int64,
            StorageData::Float32(_) => DType::Float,
            StorageData::Float64(_) => DType::Double,
            StorageData::Bool(_) => DType::Bool,
        }
    }

    /// Clone the storage (creates a new Arc reference)
    pub fn shallow_clone(&self) -> Self {
        Storage {
            data: Arc::clone(&self.data),
        }
    }

    /// Deep clone the storage (creates a new copy of data)
    pub fn deep_clone(&self) -> Self {
        let data = match &*self.data {
            StorageData::Uint8(v) => StorageData::Uint8(v.clone()),
            StorageData::Int8(v) => StorageData::Int8(v.clone()),
            StorageData::Int16(v) => StorageData::Int16(v.clone()),
            StorageData::Int32(v) => StorageData::Int32(v.clone()),
            StorageData::Int64(v) => StorageData::Int64(v.clone()),
            StorageData::Float32(v) => StorageData::Float32(v.clone()),
            StorageData::Float64(v) => StorageData::Float64(v.clone()),
            StorageData::Bool(v) => StorageData::Bool(v.clone()),
        };
        Storage {
            data: Arc::new(data),
        }
    }

    /// Get a reference to the underlying data as f32 slice
    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        match &*self.data {
            StorageData::Float32(v) => Ok(v.as_slice()),
            _ => Err(TchError::TypeError("Storage is not f32".to_string())),
        }
    }

    /// Get a reference to the underlying data as f64 slice
    pub fn as_f64_slice(&self) -> Result<&[f64]> {
        match &*self.data {
            StorageData::Float64(v) => Ok(v.as_slice()),
            _ => Err(TchError::TypeError("Storage is not f64".to_string())),
        }
    }

    /// Get a reference to the underlying data as i32 slice
    pub fn as_i32_slice(&self) -> Result<&[i32]> {
        match &*self.data {
            StorageData::Int32(v) => Ok(v.as_slice()),
            _ => Err(TchError::TypeError("Storage is not i32".to_string())),
        }
    }

    /// Get a reference to the underlying data as i64 slice
    pub fn as_i64_slice(&self) -> Result<&[i64]> {
        match &*self.data {
            StorageData::Int64(v) => Ok(v.as_slice()),
            _ => Err(TchError::TypeError("Storage is not i64".to_string())),
        }
    }

    /// Get a mutable copy of the data as a vector of f64
    pub fn to_vec_f64(&self) -> Vec<f64> {
        match &*self.data {
            StorageData::Uint8(v) => v.iter().map(|&x| x as f64).collect(),
            StorageData::Int8(v) => v.iter().map(|&x| x as f64).collect(),
            StorageData::Int16(v) => v.iter().map(|&x| x as f64).collect(),
            StorageData::Int32(v) => v.iter().map(|&x| x as f64).collect(),
            StorageData::Int64(v) => v.iter().map(|&x| x as f64).collect(),
            StorageData::Float32(v) => v.iter().map(|&x| x as f64).collect(),
            StorageData::Float64(v) => v.clone(),
            StorageData::Bool(v) => v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect(),
        }
    }
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        self.shallow_clone()
    }
}
