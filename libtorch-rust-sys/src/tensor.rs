use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Result, TchError};
use crate::scalar::Scalar;
use crate::storage::Storage;
use crate::autograd::{Edge, GradNode};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

/// Internal tensor implementation
#[derive(Clone)]
pub struct TensorImpl {
    storage: Storage,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    dtype: DType,
    device: Device,

    // Autograd fields
    /// Gradient of this tensor (accumulated during backward pass)
    grad: Option<Box<TensorImpl>>,

    /// Whether this tensor requires gradient computation
    requires_grad: bool,

    /// The gradient function that created this tensor (for backward pass)
    grad_fn: Option<Arc<dyn GradNode>>,
}

impl TensorImpl {
    /// Create a new tensor with the given shape and dtype
    pub fn new(shape: &[usize], dtype: DType, device: Device) -> Result<Self> {
        let numel = shape.iter().product();
        let storage = Storage::new(numel, dtype, device)?;
        let strides = Self::compute_strides(shape);

        Ok(TensorImpl {
            storage,
            shape: shape.to_vec(),
            strides,
            offset: 0,
            dtype,
            device,
            grad: None,
            requires_grad: false,
            grad_fn: None,
        })
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize], dtype: DType, device: Device) -> Result<Self> {
        Self::new(shape, dtype, device)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize], dtype: DType, device: Device) -> Result<Self> {
        let mut tensor = Self::new(shape, dtype, device)?;
        tensor.fill_(Scalar::Float(1.0))?;
        Ok(tensor)
    }

    /// Create a tensor from a slice of f32 values
    pub fn from_slice_f32(data: &[f32], shape: &[usize]) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(TchError::ShapeError(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                numel
            )));
        }

        let storage = Storage::from_vec_f32(data.to_vec());
        let strides = Self::compute_strides(shape);

        Ok(TensorImpl {
            storage,
            shape: shape.to_vec(),
            strides,
            offset: 0,
            dtype: DType::Float,
            device: Device::Cpu,
            grad: None,
            requires_grad: false,
            grad_fn: None,
        })
    }

    /// Create a tensor from a slice of f64 values
    pub fn from_slice_f64(data: &[f64], shape: &[usize]) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(TchError::ShapeError(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                numel
            )));
        }

        let storage = Storage::from_vec_f64(data.to_vec());
        let strides = Self::compute_strides(shape);

        Ok(TensorImpl {
            storage,
            shape: shape.to_vec(),
            strides,
            offset: 0,
            dtype: DType::Double,
            device: Device::Cpu,
            grad: None,
            requires_grad: false,
            grad_fn: None,
        })
    }

    /// Create a tensor from a slice of i64 values
    pub fn from_slice_i64(data: &[i64], shape: &[usize]) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(TchError::ShapeError(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                numel
            )));
        }

        let storage = Storage::from_vec_i64(data.to_vec());
        let strides = Self::compute_strides(shape);

        Ok(TensorImpl {
            storage,
            shape: shape.to_vec(),
            strides,
            offset: 0,
            dtype: DType::Int64,
            device: Device::Cpu,
            grad: None,
            requires_grad: false,
            grad_fn: None,
        })
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of the tensor
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get the size of a specific dimension
    pub fn size(&self, dim: usize) -> Result<usize> {
        if dim >= self.ndim() {
            return Err(TchError::IndexError(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }
        Ok(self.shape[dim])
    }

    /// Reshape the tensor
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(TchError::ShapeError(format!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                self.numel(),
                new_shape,
                new_numel
            )));
        }

        let new_strides = Self::compute_strides(new_shape);

        // For simplicity, always create a contiguous copy for reshape
        let mut result = Self::new(new_shape, self.dtype, self.device)?;
        self.copy_data_to(&mut result)?;

        Ok(result)
    }

    /// Transpose the tensor (2D only for now)
    pub fn transpose(&self) -> Result<Self> {
        if self.ndim() != 2 {
            return Err(TchError::ShapeError(
                "transpose() requires a 2D tensor".to_string(),
            ));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let data = self.to_vec_f64();

        // Rearrange data for transpose
        let mut transposed_data = Vec::with_capacity(data.len());
        for j in 0..cols {
            for i in 0..rows {
                transposed_data.push(data[i * cols + j]);
            }
        }

        let new_shape = vec![cols, rows];

        match self.dtype {
            DType::Float => {
                let f32_data: Vec<f32> = transposed_data.iter().map(|&x| x as f32).collect();
                Self::from_slice_f32(&f32_data, &new_shape)
            }
            DType::Double => Self::from_slice_f64(&transposed_data, &new_shape),
            DType::Int64 => {
                let i64_data: Vec<i64> = transposed_data.iter().map(|&x| x as i64).collect();
                Self::from_slice_i64(&i64_data, &new_shape)
            }
            _ => Err(TchError::TypeError(
                "Unsupported dtype for transpose".to_string(),
            )),
        }
    }

    /// Add a dimension of size 1 at the specified position
    pub fn unsqueeze(&self, dim: isize) -> Result<Self> {
        let ndim = self.ndim() as isize;
        let dim = if dim < 0 { ndim + dim + 1 } else { dim };

        if dim < 0 || dim > ndim {
            return Err(TchError::IndexError(format!(
                "Dimension {} out of range for unsqueeze",
                dim
            )));
        }

        let dim = dim as usize;
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);

        let mut new_strides = self.strides.clone();
        new_strides.insert(dim, 1);

        Ok(TensorImpl {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
            device: self.device,
            grad: None,
            requires_grad: self.requires_grad,
            grad_fn: None,  // TODO: Set grad_fn for unsqueeze operation
        })
    }

    /// Remove dimensions of size 1
    pub fn squeeze(&self) -> Self {
        let (new_shape, new_strides): (Vec<_>, Vec<_>) = self
            .shape
            .iter()
            .zip(self.strides.iter())
            .filter(|(&size, _)| size != 1)
            .map(|(&size, &stride)| (size, stride))
            .unzip();

        TensorImpl {
            storage: self.storage.clone(),
            shape: if new_shape.is_empty() {
                vec![1]
            } else {
                new_shape
            },
            strides: if new_strides.is_empty() {
                vec![1]
            } else {
                new_strides
            },
            offset: self.offset,
            dtype: self.dtype,
            device: self.device,
            grad: None,
            requires_grad: self.requires_grad,
            grad_fn: None,  // TODO: Set grad_fn for squeeze operation
        }
    }

    /// Fill the tensor with a scalar value
    pub fn fill_(&mut self, value: Scalar) -> Result<()> {
        // For simplicity, only support contiguous tensors for now
        if !self.is_contiguous() {
            return Err(TchError::TensorError(
                "fill_ only supports contiguous tensors".to_string(),
            ));
        }

        match self.dtype {
            DType::Float => {
                let val = value.to_f64() as f32;
                let storage = Storage::from_vec_f32(vec![val; self.numel()]);
                self.storage = storage;
            }
            DType::Double => {
                let val = value.to_f64();
                let storage = Storage::from_vec_f64(vec![val; self.numel()]);
                self.storage = storage;
            }
            DType::Int64 => {
                let val = value.to_i64();
                let storage = Storage::from_vec_i64(vec![val; self.numel()]);
                self.storage = storage;
            }
            _ => {
                return Err(TchError::TypeError(
                    "fill_ not implemented for this dtype".to_string(),
                ))
            }
        }

        Ok(())
    }

    /// Check if the tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }

        let expected_strides = Self::compute_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Create a contiguous copy of the tensor
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let mut result = Self::new(&self.shape, self.dtype, self.device)?;
        self.copy_data_to(&mut result)?;
        Ok(result)
    }

    /// Compute strides for a given shape (row-major / C-contiguous)
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() {
            return vec![];
        }

        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Copy data from this tensor to another tensor
    fn copy_data_to(&self, dst: &mut TensorImpl) -> Result<()> {
        if self.numel() != dst.numel() {
            return Err(TchError::ShapeError(format!(
                "Tensors must have the same number of elements: {} vs {}",
                self.numel(),
                dst.numel()
            )));
        }

        if self.dtype != dst.dtype {
            return Err(TchError::TypeError(
                "Tensors must have the same dtype".to_string(),
            ));
        }

        // For now, simple implementation that converts to f64 and back
        let data = self.to_vec_f64();
        match dst.dtype {
            DType::Float => {
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                dst.storage = Storage::from_vec_f32(f32_data);
            }
            DType::Double => {
                dst.storage = Storage::from_vec_f64(data);
            }
            DType::Int64 => {
                let i64_data: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                dst.storage = Storage::from_vec_i64(i64_data);
            }
            _ => {
                return Err(TchError::TypeError(
                    "Unsupported dtype for copy".to_string(),
                ))
            }
        }

        Ok(())
    }

    /// Convert tensor data to a Vec<f64>
    pub fn to_vec_f64(&self) -> Vec<f64> {
        self.storage.to_vec_f64()
    }

    /// Get data as f32 slice (if dtype is Float)
    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        if self.dtype != DType::Float {
            return Err(TchError::TypeError("Tensor is not f32".to_string()));
        }
        if !self.is_contiguous() {
            return Err(TchError::TensorError(
                "Tensor must be contiguous".to_string(),
            ));
        }
        self.storage.as_f32_slice()
    }

    /// Get data as f64 slice (if dtype is Double)
    pub fn as_f64_slice(&self) -> Result<&[f64]> {
        if self.dtype != DType::Double {
            return Err(TchError::TypeError("Tensor is not f64".to_string()));
        }
        if !self.is_contiguous() {
            return Err(TchError::TensorError(
                "Tensor must be contiguous".to_string(),
            ));
        }
        self.storage.as_f64_slice()
    }

    /// Get data as i64 slice (if dtype is Int64)
    pub fn as_i64_slice(&self) -> Result<&[i64]> {
        if self.dtype != DType::Int64 {
            return Err(TchError::TypeError("Tensor is not i64".to_string()));
        }
        if !self.is_contiguous() {
            return Err(TchError::TensorError(
                "Tensor must be contiguous".to_string(),
            ));
        }
        self.storage.as_i64_slice()
    }

    /// Element-wise addition
    pub fn add(&self, other: &TensorImpl) -> Result<TensorImpl> {
        self.binary_op(other, |a, b| a + b)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &TensorImpl) -> Result<TensorImpl> {
        self.binary_op(other, |a, b| a - b)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &TensorImpl) -> Result<TensorImpl> {
        let mut result = self.binary_op(other, |a, b| a * b)?;

        // Record operation for autograd if either input requires grad and grad is enabled
        if crate::autograd::is_grad_enabled() && (self.requires_grad || other.requires_grad) {
            use crate::autograd::ops::MulBackward;

            // Set requires_grad on result
            result.requires_grad = true;

            // Create backward node and set as grad_fn
            let mul_backward = MulBackward::new(self, other);
            result.set_grad_fn(mul_backward);
        }

        Ok(result)
    }

    /// Element-wise division
    pub fn div(&self, other: &TensorImpl) -> Result<TensorImpl> {
        self.binary_op(other, |a, b| a / b)
    }

    /// Add a scalar to the tensor
    pub fn add_scalar(&self, scalar: Scalar) -> Result<TensorImpl> {
        let val = scalar.to_f64();
        self.unary_op(|x| x + val)
    }

    /// Multiply the tensor by a scalar
    pub fn mul_scalar(&self, scalar: Scalar) -> Result<TensorImpl> {
        let val = scalar.to_f64();
        self.unary_op(|x| x * val)
    }

    /// Negate the tensor
    pub fn neg(&self) -> Result<TensorImpl> {
        self.unary_op(|x| -x)
    }

    /// Apply a binary operation element-wise with broadcasting
    fn binary_op<F>(&self, other: &TensorImpl, f: F) -> Result<TensorImpl>
    where
        F: Fn(f64, f64) -> f64,
    {
        // Check if broadcasting is needed
        if self.shape == other.shape {
            // Same shape, no broadcasting needed
            let self_data = self.to_vec_f64();
            let other_data = other.to_vec_f64();

            let result_data: Vec<f64> = self_data
                .iter()
                .zip(other_data.iter())
                .map(|(&a, &b)| f(a, b))
                .collect();

            return self.create_result_tensor(result_data, &self.shape, other);
        }

        // Simple broadcasting: allow adding a 1D tensor to the last dimension of a 2D tensor
        if self.ndim() == 2 && other.ndim() == 1 && self.shape[1] == other.shape[0] {
            let self_data = self.to_vec_f64();
            let other_data = other.to_vec_f64();

            let rows = self.shape[0];
            let cols = self.shape[1];

            let mut result_data = Vec::with_capacity(self_data.len());
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    result_data.push(f(self_data[idx], other_data[j]));
                }
            }

            return self.create_result_tensor(result_data, &self.shape, other);
        }

        // Allow broadcasting a scalar (shape [1]) to any shape
        if other.numel() == 1 {
            let self_data = self.to_vec_f64();
            let scalar = other.to_vec_f64()[0];

            let result_data: Vec<f64> = self_data.iter().map(|&a| f(a, scalar)).collect();

            return self.create_result_tensor(result_data, &self.shape, other);
        }

        if self.numel() == 1 {
            let scalar = self.to_vec_f64()[0];
            let other_data = other.to_vec_f64();

            let result_data: Vec<f64> = other_data.iter().map(|&b| f(scalar, b)).collect();

            return self.create_result_tensor(result_data, &other.shape, other);
        }

        Err(TchError::ShapeError(format!(
            "Shape mismatch: {:?} vs {:?} (broadcasting not supported for these shapes)",
            self.shape, other.shape
        )))
    }

    /// Helper to create result tensor with appropriate dtype
    fn create_result_tensor(
        &self,
        result_data: Vec<f64>,
        shape: &[usize],
        other: &TensorImpl,
    ) -> Result<TensorImpl> {
        // Result dtype is the "wider" of the two
        let result_dtype = if self.dtype == DType::Double || other.dtype == DType::Double {
            DType::Double
        } else if self.dtype == DType::Float || other.dtype == DType::Float {
            DType::Float
        } else {
            self.dtype
        };

        match result_dtype {
            DType::Float => {
                let f32_data: Vec<f32> = result_data.iter().map(|&x| x as f32).collect();
                Self::from_slice_f32(&f32_data, shape)
            }
            DType::Double => Self::from_slice_f64(&result_data, shape),
            DType::Int64 => {
                let i64_data: Vec<i64> = result_data.iter().map(|&x| x as i64).collect();
                Self::from_slice_i64(&i64_data, shape)
            }
            _ => Err(TchError::TypeError(
                "Unsupported dtype for binary op".to_string(),
            )),
        }
    }

    /// Apply a unary operation element-wise
    fn unary_op<F>(&self, f: F) -> Result<TensorImpl>
    where
        F: Fn(f64) -> f64,
    {
        let data = self.to_vec_f64();
        let result_data: Vec<f64> = data.iter().map(|&x| f(x)).collect();

        match self.dtype {
            DType::Float => {
                let f32_data: Vec<f32> = result_data.iter().map(|&x| x as f32).collect();
                Self::from_slice_f32(&f32_data, &self.shape)
            }
            DType::Double => Self::from_slice_f64(&result_data, &self.shape),
            DType::Int64 => {
                let i64_data: Vec<i64> = result_data.iter().map(|&x| x as i64).collect();
                Self::from_slice_i64(&i64_data, &self.shape)
            }
            _ => Err(TchError::TypeError(
                "Unsupported dtype for unary op".to_string(),
            )),
        }
    }

    /// Matrix multiplication (2D only for now)
    pub fn matmul(&self, other: &TensorImpl) -> Result<TensorImpl> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(TchError::ShapeError(
                "matmul requires 2D tensors".to_string(),
            ));
        }

        let m = self.shape[0];
        let k = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];

        if k != k2 {
            return Err(TchError::ShapeError(format!(
                "Matrix dimension mismatch: ({}, {}) x ({}, {})",
                m, k, k2, n
            )));
        }

        let self_data = self.to_vec_f64();
        let other_data = other.to_vec_f64();

        let mut result_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self_data[i * k + p] * other_data[p * n + j];
                }
                result_data[i * n + j] = sum;
            }
        }

        let result_dtype = if self.dtype == DType::Double || other.dtype == DType::Double {
            DType::Double
        } else {
            DType::Float
        };

        match result_dtype {
            DType::Float => {
                let f32_data: Vec<f32> = result_data.iter().map(|&x| x as f32).collect();
                Self::from_slice_f32(&f32_data, &[m, n])
            }
            DType::Double => Self::from_slice_f64(&result_data, &[m, n]),
            _ => Err(TchError::TypeError(
                "Unsupported dtype for matmul".to_string(),
            )),
        }
    }

    // ============================================================
    // Autograd methods (Phase 1.1.2)
    // ============================================================

    /// Set whether this tensor requires gradient computation
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Check if this tensor requires gradient computation
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get the gradient of this tensor (if it exists)
    pub fn grad(&self) -> Option<&TensorImpl> {
        self.grad.as_ref().map(|boxed| boxed.as_ref())
    }

    /// Get a mutable reference to the gradient (if it exists)
    pub fn grad_mut(&mut self) -> Option<&mut TensorImpl> {
        self.grad.as_mut().map(|boxed| boxed.as_mut())
    }

    /// Set the gradient of this tensor
    pub fn set_grad(&mut self, gradient: TensorImpl) {
        self.grad = Some(Box::new(gradient));
    }

    /// Clear the gradient of this tensor
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Get the gradient function that created this tensor
    pub fn grad_fn(&self) -> Option<&Arc<dyn GradNode>> {
        self.grad_fn.as_ref()
    }

    /// Set the gradient function for this tensor
    pub fn set_grad_fn(&mut self, grad_fn: Arc<dyn GradNode>) {
        self.grad_fn = Some(grad_fn);
    }

    // ============================================================
    // Gradient accumulation (Phase 1.1.3)
    // ============================================================

    /// Accumulate gradient (add new_grad to existing gradient)
    ///
    /// If no gradient exists yet, this becomes the first gradient.
    /// Otherwise, the new gradient is added to the existing one.
    ///
    /// # Arguments
    /// * `new_grad` - The gradient to accumulate
    ///
    /// # Returns
    /// Ok(()) on success, or an error if shapes don't match
    pub fn accumulate_grad(&mut self, new_grad: TensorImpl) -> Result<()> {
        if self.shape != new_grad.shape {
            return Err(TchError::ShapeError(format!(
                "Gradient shape {:?} doesn't match tensor shape {:?}",
                new_grad.shape, self.shape
            )));
        }

        match &mut self.grad {
            None => {
                // First gradient - just store it
                self.grad = Some(Box::new(new_grad));
                Ok(())
            }
            Some(existing_grad) => {
                // Accumulate by adding
                let accumulated = existing_grad.add(&new_grad)?;
                **existing_grad = accumulated;
                Ok(())
            }
        }
    }
}
