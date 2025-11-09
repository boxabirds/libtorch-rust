use std::fmt;

/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 8-bit unsigned integer
    Uint8,
    /// 8-bit signed integer
    Int8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int,
    /// 64-bit signed integer
    Int64,
    /// 16-bit floating point (half precision)
    Half,
    /// 32-bit floating point (single precision)
    Float,
    /// 64-bit floating point (double precision)
    Double,
    /// Complex 32-bit floating point
    ComplexHalf,
    /// Complex 64-bit floating point
    ComplexFloat,
    /// Complex 128-bit floating point
    ComplexDouble,
    /// Boolean
    Bool,
    /// BFloat16
    BFloat16,
}

impl DType {
    /// Returns the size in bytes of this data type
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::Uint8 | DType::Int8 | DType::Bool => 1,
            DType::Int16 | DType::Half | DType::BFloat16 => 2,
            DType::Int | DType::Float | DType::ComplexHalf => 4,
            DType::Int64 | DType::Double | DType::ComplexFloat => 8,
            DType::ComplexDouble => 16,
        }
    }

    /// Returns true if this is a floating point type
    pub fn is_floating_point(&self) -> bool {
        matches!(
            self,
            DType::Half
                | DType::Float
                | DType::Double
                | DType::BFloat16
                | DType::ComplexHalf
                | DType::ComplexFloat
                | DType::ComplexDouble
        )
    }

    /// Returns true if this is a complex type
    pub fn is_complex(&self) -> bool {
        matches!(
            self,
            DType::ComplexHalf | DType::ComplexFloat | DType::ComplexDouble
        )
    }

    /// Returns true if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DType::Uint8 | DType::Int8 | DType::Int16 | DType::Int | DType::Int64
        )
    }
}

impl Default for DType {
    fn default() -> Self {
        DType::Float
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Uint8 => write!(f, "uint8"),
            DType::Int8 => write!(f, "int8"),
            DType::Int16 => write!(f, "int16"),
            DType::Int => write!(f, "int32"),
            DType::Int64 => write!(f, "int64"),
            DType::Half => write!(f, "float16"),
            DType::Float => write!(f, "float32"),
            DType::Double => write!(f, "float64"),
            DType::ComplexHalf => write!(f, "complex32"),
            DType::ComplexFloat => write!(f, "complex64"),
            DType::ComplexDouble => write!(f, "complex128"),
            DType::Bool => write!(f, "bool"),
            DType::BFloat16 => write!(f, "bfloat16"),
        }
    }
}
