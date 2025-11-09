use crate::{Device, Kind};
use libtorch_rust_sys::{Scalar, TchError, TensorImpl};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Tensor type compatible with tch-rs API
#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: TensorImpl,
}

impl Tensor {
    /// Create a new tensor from internal implementation
    pub(crate) fn from_impl(inner: TensorImpl) -> Self {
        Tensor { inner }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[i64], kind: Kind, device: Device) -> Self {
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let inner = TensorImpl::zeros(&shape_usize, kind.to_dtype(), device)
            .expect("Failed to create zeros tensor");
        Tensor { inner }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[i64], kind: Kind, device: Device) -> Self {
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let inner = TensorImpl::ones(&shape_usize, kind.to_dtype(), device)
            .expect("Failed to create ones tensor");
        Tensor { inner }
    }

    /// Create a 1D tensor from a slice
    pub fn of_slice(data: &[f64]) -> Self {
        let inner = TensorImpl::from_slice_f64(data, &[data.len()])
            .expect("Failed to create tensor from slice");
        Tensor { inner }
    }

    /// Create a tensor from a slice with a specific shape
    pub fn of_data_size(data: &[f64], shape: &[i64]) -> Self {
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let inner = TensorImpl::from_slice_f64(data, &shape_usize)
            .expect("Failed to create tensor from data and size");
        Tensor { inner }
    }

    /// Create a tensor from f32 slice
    pub fn from_slice(data: &[f32]) -> Self {
        let inner = TensorImpl::from_slice_f32(data, &[data.len()])
            .expect("Failed to create tensor from f32 slice");
        Tensor { inner }
    }

    /// Create a 2D tensor from f32 slice
    pub fn from_slice2(data: &[f32], rows: i64, cols: i64) -> Self {
        let shape = vec![rows as usize, cols as usize];
        let inner = TensorImpl::from_slice_f32(data, &shape)
            .expect("Failed to create 2D tensor from slice");
        Tensor { inner }
    }

    /// Get the shape of the tensor
    pub fn size(&self) -> Vec<i64> {
        self.inner.shape().iter().map(|&x| x as i64).collect()
    }

    /// Get the size of a specific dimension
    pub fn size1(&self, dim: i64) -> Result<i64, TchError> {
        let dim_usize = if dim < 0 {
            (self.inner.ndim() as i64 + dim) as usize
        } else {
            dim as usize
        };
        self.inner.size(dim_usize).map(|x| x as i64)
    }

    /// Get the number of dimensions
    pub fn dim(&self) -> i64 {
        self.inner.ndim() as i64
    }

    /// Get the total number of elements
    pub fn numel(&self) -> i64 {
        self.inner.numel() as i64
    }

    /// Get the data type (kind)
    pub fn kind(&self) -> Kind {
        Kind::from_dtype(self.inner.dtype())
    }

    /// Get the device
    pub fn device(&self) -> Device {
        self.inner.device()
    }

    /// Reshape the tensor
    pub fn reshape(&self, shape: &[i64]) -> Self {
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let inner = self.inner.reshape(&shape_usize)
            .expect("Failed to reshape tensor");
        Tensor { inner }
    }

    /// View the tensor with a new shape (alias for reshape)
    pub fn view(&self, shape: &[i64]) -> Self {
        self.reshape(shape)
    }

    /// Transpose a 2D tensor
    pub fn transpose(&self, dim0: i64, dim1: i64) -> Self {
        // For now, only support 2D transpose
        if self.dim() != 2 || dim0 != 0 || dim1 != 1 {
            panic!("Only 2D transpose with dims (0, 1) is currently supported");
        }
        let inner = self.inner.transpose()
            .expect("Failed to transpose tensor");
        Tensor { inner }
    }

    /// Transpose (shorthand for transpose(0, 1))
    pub fn tr(&self) -> Self {
        self.transpose(0, 1)
    }

    /// Add a dimension of size 1
    pub fn unsqueeze(&self, dim: i64) -> Self {
        let inner = self.inner.unsqueeze(dim as isize)
            .expect("Failed to unsqueeze tensor");
        Tensor { inner }
    }

    /// Remove dimensions of size 1
    pub fn squeeze(&self) -> Self {
        let inner = self.inner.squeeze();
        Tensor { inner }
    }

    /// Remove a specific dimension of size 1
    pub fn squeeze_dim(&self, dim: i64) -> Self {
        // TODO: Implement squeeze for specific dimension
        // For now, just squeeze all
        let _ = dim;
        self.squeeze()
    }

    /// Create a contiguous copy of the tensor
    pub fn contiguous(&self) -> Self {
        let inner = self.inner.contiguous()
            .expect("Failed to make tensor contiguous");
        Tensor { inner }
    }

    /// Check if the tensor is contiguous
    pub fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    /// Clone the tensor (creates a deep copy)
    pub fn shallow_clone(&self) -> Self {
        Tensor {
            inner: self.inner.clone(),
        }
    }

    /// Copy data from another tensor
    pub fn copy_(&mut self, src: &Tensor) -> &mut Self {
        // TODO: Implement proper copy_
        self.inner = src.inner.clone();
        self
    }

    /// Fill the tensor with a scalar value
    pub fn fill_(&mut self, value: f64) -> &mut Self {
        self.inner.fill_(Scalar::Float(value))
            .expect("Failed to fill tensor");
        self
    }

    /// Zero out the tensor
    pub fn zero_(&mut self) -> &mut Self {
        self.fill_(0.0)
    }

    /// Convert tensor data to Vec<f64>
    pub fn to_vec_f64(&self) -> Vec<f64> {
        self.inner.to_vec_f64()
    }

    /// Get a reference to the underlying f32 data
    pub fn try_as_f32_slice(&self) -> Result<&[f32], TchError> {
        self.inner.as_f32_slice()
    }

    /// Get a reference to the underlying f64 data
    pub fn try_as_f64_slice(&self) -> Result<&[f64], TchError> {
        self.inner.as_f64_slice()
    }

    /// Get a reference to the underlying i64 data
    pub fn try_as_i64_slice(&self) -> Result<&[i64], TchError> {
        self.inner.as_i64_slice()
    }

    /// Element-wise addition with another tensor
    pub fn add_tensor(&self, other: &Tensor) -> Self {
        let inner = self.inner.add(&other.inner)
            .expect("Failed to add tensors");
        Tensor { inner }
    }

    /// Element-wise subtraction with another tensor
    pub fn sub_tensor(&self, other: &Tensor) -> Self {
        let inner = self.inner.sub(&other.inner)
            .expect("Failed to subtract tensors");
        Tensor { inner }
    }

    /// Element-wise multiplication with another tensor
    pub fn mul_tensor(&self, other: &Tensor) -> Self {
        let inner = self.inner.mul(&other.inner)
            .expect("Failed to multiply tensors");
        Tensor { inner }
    }

    /// Element-wise division with another tensor
    pub fn div_tensor(&self, other: &Tensor) -> Self {
        let inner = self.inner.div(&other.inner)
            .expect("Failed to divide tensors");
        Tensor { inner }
    }

    /// Add a scalar to the tensor
    pub fn add_scalar(&self, value: f64) -> Self {
        let inner = self.inner.add_scalar(Scalar::Float(value))
            .expect("Failed to add scalar");
        Tensor { inner }
    }

    /// Multiply the tensor by a scalar
    pub fn mul_scalar(&self, value: f64) -> Self {
        let inner = self.inner.mul_scalar(Scalar::Float(value))
            .expect("Failed to multiply by scalar");
        Tensor { inner }
    }

    /// Negate the tensor
    pub fn neg(&self) -> Self {
        let inner = self.inner.neg()
            .expect("Failed to negate tensor");
        Tensor { inner }
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Self {
        let inner = self.inner.matmul(&other.inner)
            .expect("Failed to perform matmul");
        Tensor { inner }
    }

    /// Matrix-matrix multiplication (alias for matmul)
    pub fn mm(&self, other: &Tensor) -> Self {
        self.matmul(other)
    }

    /// ReLU activation function
    pub fn relu(&self) -> Self {
        let data = self.inner.to_vec_f64();
        let result: Vec<f64> = data.iter().map(|&x| x.max(0.0)).collect();

        let shape_usize: Vec<usize> = self.inner.shape().to_vec();
        let inner = TensorImpl::from_slice_f64(&result, &shape_usize)
            .expect("Failed to create relu result");
        Tensor { inner }
    }

    /// Sigmoid activation function
    pub fn sigmoid(&self) -> Self {
        let data = self.inner.to_vec_f64();
        let result: Vec<f64> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();

        let shape_usize: Vec<usize> = self.inner.shape().to_vec();
        let inner = TensorImpl::from_slice_f64(&result, &shape_usize)
            .expect("Failed to create sigmoid result");
        Tensor { inner }
    }

    /// Softmax function
    pub fn softmax(&self, dim: i64, _dtype: Kind) -> Self {
        if self.dim() != 2 {
            panic!("softmax currently only supports 2D tensors");
        }

        let data = self.inner.to_vec_f64();
        let shape = self.inner.shape();

        let mut result = vec![0.0; data.len()];

        if dim == 1 {
            // Softmax along the last dimension
            for i in 0..shape[0] {
                let row_start = i * shape[1];
                let row = &data[row_start..row_start + shape[1]];

                // Find max for numerical stability
                let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                // Compute exp(x - max)
                let exp_vals: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let sum: f64 = exp_vals.iter().sum();

                // Normalize
                for (j, &exp_val) in exp_vals.iter().enumerate() {
                    result[row_start + j] = exp_val / sum;
                }
            }
        } else {
            panic!("softmax along dim {} not yet implemented", dim);
        }

        let shape_usize: Vec<usize> = self.inner.shape().to_vec();
        let inner = TensorImpl::from_slice_f64(&result, &shape_usize)
            .expect("Failed to create softmax result");
        Tensor { inner }
    }

    /// Sum all elements
    pub fn sum(&self, dtype: Kind) -> Self {
        let data = self.inner.to_vec_f64();
        let sum: f64 = data.iter().sum();

        let inner = TensorImpl::from_slice_f64(&[sum], &[1])
            .expect("Failed to create sum result");
        Tensor { inner }
    }

    /// Mean of all elements
    pub fn mean(&self, dtype: Kind) -> Self {
        let data = self.inner.to_vec_f64();
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;

        let inner = TensorImpl::from_slice_f64(&[mean], &[1])
            .expect("Failed to create mean result");
        Tensor { inner }
    }

    /// Convert tensor to a scalar value
    pub fn double_value(&self, indices: &[i64]) -> f64 {
        if indices.is_empty() && self.numel() == 1 {
            self.inner.to_vec_f64()[0]
        } else {
            // TODO: Implement proper indexing
            panic!("Indexing not yet fully implemented");
        }
    }

    /// Convert a scalar tensor to f64
    pub fn to_f64(&self) -> f64 {
        assert_eq!(self.numel(), 1, "Tensor must have exactly one element");
        self.inner.to_vec_f64()[0]
    }

    /// Convert a scalar tensor to i64
    pub fn to_i64(&self) -> i64 {
        assert_eq!(self.numel(), 1, "Tensor must have exactly one element");
        self.inner.to_vec_f64()[0] as i64
    }

    /// Print the tensor
    pub fn print(&self) {
        println!("Tensor(shape={:?}, dtype={:?})", self.size(), self.kind());
        if self.numel() <= 100 {
            println!("  data: {:?}", self.to_vec_f64());
        }
    }
}

// Operator overloading
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        self.add_tensor(other)
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        self.sub_tensor(other)
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        self.mul_tensor(other)
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Tensor {
        self.div_tensor(other)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self.neg()
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;

    fn add(self, scalar: f64) -> Tensor {
        self.add_scalar(scalar)
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f64) -> Tensor {
        self.mul_scalar(scalar)
    }
}
