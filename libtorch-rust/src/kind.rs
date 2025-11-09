use libtorch_rust_sys::DType;

/// Tensor element type (Kind in tch-rs terminology)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Uint8,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    BFloat16,
}

impl Kind {
    /// Convert to internal DType
    pub(crate) fn to_dtype(self) -> DType {
        match self {
            Kind::Uint8 => DType::Uint8,
            Kind::Int8 => DType::Int8,
            Kind::Int16 => DType::Int16,
            Kind::Int => DType::Int,
            Kind::Int64 => DType::Int64,
            Kind::Half => DType::Half,
            Kind::Float => DType::Float,
            Kind::Double => DType::Double,
            Kind::ComplexHalf => DType::ComplexHalf,
            Kind::ComplexFloat => DType::ComplexFloat,
            Kind::ComplexDouble => DType::ComplexDouble,
            Kind::Bool => DType::Bool,
            Kind::BFloat16 => DType::BFloat16,
        }
    }

    /// Convert from internal DType
    pub(crate) fn from_dtype(dtype: DType) -> Self {
        match dtype {
            DType::Uint8 => Kind::Uint8,
            DType::Int8 => Kind::Int8,
            DType::Int16 => Kind::Int16,
            DType::Int => Kind::Int,
            DType::Int64 => Kind::Int64,
            DType::Half => Kind::Half,
            DType::Float => Kind::Float,
            DType::Double => Kind::Double,
            DType::ComplexHalf => Kind::ComplexHalf,
            DType::ComplexFloat => Kind::ComplexFloat,
            DType::ComplexDouble => Kind::ComplexDouble,
            DType::Bool => Kind::Bool,
            DType::BFloat16 => Kind::BFloat16,
        }
    }
}

// Convenient constants
impl Kind {
    pub const UINT8: Kind = Kind::Uint8;
    pub const INT8: Kind = Kind::Int8;
    pub const INT16: Kind = Kind::Int16;
    pub const INT: Kind = Kind::Int;
    pub const INT64: Kind = Kind::Int64;
    pub const HALF: Kind = Kind::Half;
    pub const FLOAT: Kind = Kind::Float;
    pub const DOUBLE: Kind = Kind::Double;
    pub const COMPLEX_HALF: Kind = Kind::ComplexHalf;
    pub const COMPLEX_FLOAT: Kind = Kind::ComplexFloat;
    pub const COMPLEX_DOUBLE: Kind = Kind::ComplexDouble;
    pub const BOOL: Kind = Kind::Bool;
    pub const BFLOAT16: Kind = Kind::BFloat16;
}
