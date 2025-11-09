use crate::dtype::DType;
use std::fmt;

/// Scalar value that can be used in tensor operations
#[derive(Debug, Clone, Copy)]
pub enum Scalar {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl Scalar {
    /// Convert to f64
    pub fn to_f64(&self) -> f64 {
        match self {
            Scalar::Int(v) => *v as f64,
            Scalar::Float(v) => *v,
            Scalar::Bool(v) => if *v { 1.0 } else { 0.0 },
        }
    }

    /// Convert to i64
    pub fn to_i64(&self) -> i64 {
        match self {
            Scalar::Int(v) => *v,
            Scalar::Float(v) => *v as i64,
            Scalar::Bool(v) => if *v { 1 } else { 0 },
        }
    }

    /// Convert to bool
    pub fn to_bool(&self) -> bool {
        match self {
            Scalar::Int(v) => *v != 0,
            Scalar::Float(v) => *v != 0.0,
            Scalar::Bool(v) => *v,
        }
    }

    /// Get the dtype that best represents this scalar
    pub fn dtype(&self) -> DType {
        match self {
            Scalar::Int(_) => DType::Int64,
            Scalar::Float(_) => DType::Double,
            Scalar::Bool(_) => DType::Bool,
        }
    }
}

impl From<i32> for Scalar {
    fn from(v: i32) -> Self {
        Scalar::Int(v as i64)
    }
}

impl From<i64> for Scalar {
    fn from(v: i64) -> Self {
        Scalar::Int(v)
    }
}

impl From<f32> for Scalar {
    fn from(v: f32) -> Self {
        Scalar::Float(v as f64)
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Scalar::Float(v)
    }
}

impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Scalar::Bool(v)
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Int(v) => write!(f, "{}", v),
            Scalar::Float(v) => write!(f, "{}", v),
            Scalar::Bool(v) => write!(f, "{}", v),
        }
    }
}
