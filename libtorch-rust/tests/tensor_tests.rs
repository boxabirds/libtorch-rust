// Tests ported from PyTorch's test/cpp/api/tensor.cpp

use approx::assert_relative_eq;
use tch::{Device, Kind, Tensor};

#[test]
fn test_tensor_creation_zeros() {
    let t = Tensor::zeros(&[2, 3], Kind::Float, Device::Cpu);
    assert_eq!(t.size(), vec![2, 3]);
    assert_eq!(t.numel(), 6);

    let data = t.to_vec_f64();
    assert_eq!(data.len(), 6);
    for val in data {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_tensor_creation_ones() {
    let t = Tensor::ones(&[2, 3], Kind::Float, Device::Cpu);
    assert_eq!(t.size(), vec![2, 3]);
    assert_eq!(t.numel(), 6);

    let data = t.to_vec_f64();
    assert_eq!(data.len(), 6);
    for val in data {
        assert_eq!(val, 1.0);
    }
}

#[test]
fn test_tensor_from_slice() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::of_slice(&data);

    assert_eq!(t.size(), vec![6]);
    assert_eq!(t.numel(), 6);

    let result = t.to_vec_f64();
    for (a, b) in data.iter().zip(result.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-6);
    }
}

#[test]
fn test_tensor_reshape() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::of_slice(&data);

    let t2 = t.reshape(&[2, 3]);
    assert_eq!(t2.size(), vec![2, 3]);
    assert_eq!(t2.numel(), 6);

    let t3 = t2.reshape(&[3, 2]);
    assert_eq!(t3.size(), vec![3, 2]);
    assert_eq!(t3.numel(), 6);
}

#[test]
fn test_tensor_transpose() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::of_data_size(&data, &[2, 3]);

    let t_tr = t.transpose(0, 1);
    assert_eq!(t_tr.size(), vec![3, 2]);

    // Original: [[1, 2, 3], [4, 5, 6]]
    // Transposed: [[1, 4], [2, 5], [3, 6]]
    let result = t_tr.to_vec_f64();
    assert_relative_eq!(result[0], 1.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 4.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], 2.0, epsilon = 1e-6);
    assert_relative_eq!(result[3], 5.0, epsilon = 1e-6);
    assert_relative_eq!(result[4], 3.0, epsilon = 1e-6);
    assert_relative_eq!(result[5], 6.0, epsilon = 1e-6);
}

#[test]
fn test_tensor_unsqueeze() {
    let t = Tensor::of_slice(&[1.0, 2.0, 3.0]);
    assert_eq!(t.size(), vec![3]);

    let t2 = t.unsqueeze(0);
    assert_eq!(t2.size(), vec![1, 3]);

    let t3 = t.unsqueeze(1);
    assert_eq!(t3.size(), vec![3, 1]);

    let t4 = t.unsqueeze(-1);
    assert_eq!(t4.size(), vec![3, 1]);
}

#[test]
fn test_tensor_squeeze() {
    let t = Tensor::zeros(&[1, 3, 1, 4], Kind::Float, Device::Cpu);
    assert_eq!(t.size(), vec![1, 3, 1, 4]);

    let t2 = t.squeeze();
    assert_eq!(t2.size(), vec![3, 4]);
}

#[test]
fn test_tensor_addition() {
    let t1 = Tensor::of_slice(&[1.0, 2.0, 3.0]);
    let t2 = Tensor::of_slice(&[4.0, 5.0, 6.0]);

    let t3 = &t1 + &t2;

    let result = t3.to_vec_f64();
    assert_relative_eq!(result[0], 5.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 7.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], 9.0, epsilon = 1e-6);
}

#[test]
fn test_tensor_subtraction() {
    let t1 = Tensor::of_slice(&[4.0, 5.0, 6.0]);
    let t2 = Tensor::of_slice(&[1.0, 2.0, 3.0]);

    let t3 = &t1 - &t2;

    let result = t3.to_vec_f64();
    assert_relative_eq!(result[0], 3.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 3.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], 3.0, epsilon = 1e-6);
}

#[test]
fn test_tensor_multiplication() {
    let t1 = Tensor::of_slice(&[1.0, 2.0, 3.0]);
    let t2 = Tensor::of_slice(&[2.0, 3.0, 4.0]);

    let t3 = &t1 * &t2;

    let result = t3.to_vec_f64();
    assert_relative_eq!(result[0], 2.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 6.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], 12.0, epsilon = 1e-6);
}

#[test]
fn test_tensor_division() {
    let t1 = Tensor::of_slice(&[6.0, 8.0, 12.0]);
    let t2 = Tensor::of_slice(&[2.0, 4.0, 3.0]);

    let t3 = &t1 / &t2;

    let result = t3.to_vec_f64();
    assert_relative_eq!(result[0], 3.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 2.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], 4.0, epsilon = 1e-6);
}

#[test]
fn test_tensor_scalar_operations() {
    let t = Tensor::of_slice(&[1.0, 2.0, 3.0]);

    let t_add = &t + 10.0;
    let result = t_add.to_vec_f64();
    assert_relative_eq!(result[0], 11.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 12.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], 13.0, epsilon = 1e-6);

    let t_mul = &t * 2.0;
    let result = t_mul.to_vec_f64();
    assert_relative_eq!(result[0], 2.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 4.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], 6.0, epsilon = 1e-6);
}

#[test]
fn test_tensor_negation() {
    let t = Tensor::of_slice(&[1.0, -2.0, 3.0]);
    let t_neg = -&t;

    let result = t_neg.to_vec_f64();
    assert_relative_eq!(result[0], -1.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 2.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], -3.0, epsilon = 1e-6);
}

#[test]
fn test_tensor_matmul() {
    // 2x3 matrix
    let a = Tensor::of_data_size(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // 3x2 matrix
    let b = Tensor::of_data_size(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    let c = a.matmul(&b);
    assert_eq!(c.size(), vec![2, 2]);

    // Expected result:
    // [[1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6],
    //  [4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6]]
    // = [[22, 28], [49, 64]]
    let result = c.to_vec_f64();
    assert_relative_eq!(result[0], 22.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 28.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], 49.0, epsilon = 1e-6);
    assert_relative_eq!(result[3], 64.0, epsilon = 1e-6);
}

#[test]
fn test_tensor_relu() {
    let t = Tensor::of_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let t_relu = t.relu();

    let result = t_relu.to_vec_f64();
    assert_relative_eq!(result[0], 0.0, epsilon = 1e-6);
    assert_relative_eq!(result[1], 0.0, epsilon = 1e-6);
    assert_relative_eq!(result[2], 0.0, epsilon = 1e-6);
    assert_relative_eq!(result[3], 1.0, epsilon = 1e-6);
    assert_relative_eq!(result[4], 2.0, epsilon = 1e-6);
}

#[test]
fn test_tensor_sigmoid() {
    let t = Tensor::of_slice(&[0.0]);
    let t_sig = t.sigmoid();

    let result = t_sig.to_vec_f64();
    assert_relative_eq!(result[0], 0.5, epsilon = 1e-6);

    // Test with larger values
    let t2 = Tensor::of_slice(&[-100.0, 0.0, 100.0]);
    let t2_sig = t2.sigmoid();
    let result2 = t2_sig.to_vec_f64();

    assert!(result2[0] < 0.01); // Should be very close to 0
    assert_relative_eq!(result2[1], 0.5, epsilon = 1e-6);
    assert!(result2[2] > 0.99); // Should be very close to 1
}

#[test]
fn test_tensor_sum() {
    let t = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let sum = t.sum(Kind::Float);

    assert_relative_eq!(sum.to_f64(), 15.0, epsilon = 1e-6);
}

#[test]
fn test_tensor_mean() {
    let t = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let mean = t.mean(Kind::Float);

    assert_relative_eq!(mean.to_f64(), 3.0, epsilon = 1e-6);
}

#[test]
fn test_tensor_contiguous() {
    let t = Tensor::zeros(&[2, 3], Kind::Float, Device::Cpu);
    assert!(t.is_contiguous());

    let t_cont = t.contiguous();
    assert!(t_cont.is_contiguous());
}

#[test]
fn test_tensor_device() {
    let t = Tensor::zeros(&[2, 3], Kind::Float, Device::Cpu);
    assert_eq!(t.device(), Device::Cpu);
}

#[test]
fn test_tensor_kind() {
    let t = Tensor::zeros(&[2, 3], Kind::Float, Device::Cpu);
    assert_eq!(t.kind(), Kind::Float);
}
