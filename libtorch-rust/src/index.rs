/// Tensor indexing operations

/// Index type for tensor slicing
#[derive(Debug, Clone)]
pub enum TensorIndex {
    /// Select a specific index
    Select(i64),
    /// Slice with start, end, and step
    Slice {
        start: Option<i64>,
        end: Option<i64>,
        step: i64,
    },
    /// Select all elements in this dimension
    Full,
    /// Insert a new axis
    NewAxis,
}

/// Convenient constructors
impl TensorIndex {
    pub fn narrow(start: i64, length: i64) -> Self {
        TensorIndex::Slice {
            start: Some(start),
            end: Some(start + length),
            step: 1,
        }
    }
}
