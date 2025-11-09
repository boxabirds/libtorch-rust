/// Wrapper types and utilities for compatibility

use crate::{Device, Kind, Tensor};

/// IValue type for JIT operations
#[derive(Clone)]
pub enum IValue {
    Tensor(Tensor),
    Double(f64),
    Int(i64),
    Bool(bool),
    String(String),
    Tuple(Vec<IValue>),
    List(Vec<IValue>),
    None,
}

impl From<Tensor> for IValue {
    fn from(t: Tensor) -> Self {
        IValue::Tensor(t)
    }
}

impl From<f64> for IValue {
    fn from(v: f64) -> Self {
        IValue::Double(v)
    }
}

impl From<i64> for IValue {
    fn from(v: i64) -> Self {
        IValue::Int(v)
    }
}

impl From<bool> for IValue {
    fn from(v: bool) -> Self {
        IValue::Bool(v)
    }
}

/// Scalar type (re-export from sys)
pub use libtorch_rust_sys::Scalar;
