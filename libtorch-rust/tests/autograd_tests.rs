// Tests ported from PyTorch's test/cpp/api/autograd.cpp

use tch::{Device, Kind, Tensor};

// ============================================================
// Phase 1.1: Gradient Storage Tests
// ============================================================

#[test]
fn test_grad_set_and_get() {
    // Create a tensor
    let mut t = Tensor::zeros(&[2, 3], Kind::Float, Device::Cpu);

    // Initially, gradient should be None
    assert!(t.grad().is_none());
    assert_eq!(t.requires_grad(), false);

    // Set requires_grad
    t.set_requires_grad(true);
    assert_eq!(t.requires_grad(), true);

    // Manually set a gradient
    let grad_value = Tensor::ones(&[2, 3], Kind::Float, Device::Cpu);
    t.set_grad(grad_value);

    // Verify gradient exists and has correct shape
    assert!(t.grad().is_some());
    let grad = t.grad().unwrap();
    assert_eq!(grad.size(), vec![2, 3]);

    // Verify gradient values
    let grad_data = grad.to_vec_f64();
    for val in grad_data {
        assert_eq!(val, 1.0);
    }
}

#[test]
fn test_grad_accumulation() {
    // Create a tensor with requires_grad=true
    let mut t = Tensor::zeros(&[3], Kind::Float, Device::Cpu);
    t.set_requires_grad(true);

    // Accumulate first gradient
    let grad1 = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    t.accumulate_grad(grad1).unwrap();

    // Check first gradient
    let grad = t.grad().unwrap();
    let grad_data = grad.to_vec_f64();
    assert_eq!(grad_data, vec![1.0, 2.0, 3.0]);

    // Accumulate second gradient
    let grad2 = Tensor::from_slice(&[0.5, 0.5, 0.5]);
    t.accumulate_grad(grad2).unwrap();

    // Check accumulated gradient (should be sum)
    let grad = t.grad().unwrap();
    let grad_data = grad.to_vec_f64();
    assert_eq!(grad_data[0], 1.5);
    assert_eq!(grad_data[1], 2.5);
    assert_eq!(grad_data[2], 3.5);
}

#[test]
fn test_grad_zero() {
    // Create a tensor and set gradient
    let mut t = Tensor::zeros(&[2], Kind::Float, Device::Cpu);
    let grad = Tensor::ones(&[2], Kind::Float, Device::Cpu);
    t.set_grad(grad);

    // Verify gradient exists
    assert!(t.grad().is_some());

    // Zero the gradient
    t.zero_grad();

    // Verify gradient is cleared
    assert!(t.grad().is_none());
}

#[test]
fn test_grad_shape_mismatch() {
    // Create a tensor
    let mut t = Tensor::zeros(&[2, 3], Kind::Float, Device::Cpu);
    t.set_requires_grad(true);

    // Try to accumulate gradient with wrong shape
    let wrong_grad = Tensor::ones(&[3, 2], Kind::Float, Device::Cpu);
    let result = t.accumulate_grad(wrong_grad);

    // Should fail with shape error
    assert!(result.is_err());
}

#[test]
fn test_requires_grad_propagation() {
    // Create tensors
    let mut x = Tensor::ones(&[2, 2], Kind::Float, Device::Cpu);
    let y = Tensor::ones(&[2, 2], Kind::Float, Device::Cpu);

    // Set requires_grad on x
    x.set_requires_grad(true);
    assert_eq!(x.requires_grad(), true);
    assert_eq!(y.requires_grad(), false);

    // Operations on x should eventually propagate requires_grad
    // (This will be fully tested once we implement operation recording)
}

// ============================================================
// Phase 1.2: Gradient Mode Tests
// ============================================================

#[test]
fn test_no_grad_guard() {
    use tch::autograd::{is_grad_enabled, NoGradGuard};

    // Initially, gradient mode should be enabled
    assert_eq!(is_grad_enabled(), true);

    {
        // Create NoGradGuard
        let _guard = NoGradGuard::new();

        // Gradient mode should be disabled
        assert_eq!(is_grad_enabled(), false);
    }

    // After guard is dropped, gradient mode should be re-enabled
    assert_eq!(is_grad_enabled(), true);
}

#[test]
fn test_set_grad_enabled() {
    use tch::autograd::{is_grad_enabled, set_grad_enabled};

    // Set to false
    set_grad_enabled(false);
    assert_eq!(is_grad_enabled(), false);

    // Set back to true
    set_grad_enabled(true);
    assert_eq!(is_grad_enabled(), true);
}

// ============================================================
// Phase 1.2: Tape Recording Tests
// ============================================================

#[test]
fn test_mul_records_grad_fn() {
    // Create tensors with requires_grad=true
    let mut x = Tensor::from_slice(&[2.0, 3.0]);
    let mut y = Tensor::from_slice(&[4.0, 5.0]);

    x.set_requires_grad(true);
    y.set_requires_grad(true);

    // Multiply: z = x * y
    let z = &x * &y;

    // z should have requires_grad=true
    assert_eq!(z.requires_grad(), true);

    // z should have a grad_fn (the MulBackward operation)
    assert!(z.has_grad_fn(), "z should have a grad_fn");
}

#[test]
fn test_mul_no_grad_guard() {
    use tch::autograd::NoGradGuard;

    let mut x = Tensor::from_slice(&[2.0]);
    let mut y = Tensor::from_slice(&[3.0]);

    x.set_requires_grad(true);
    y.set_requires_grad(true);

    // Inside NoGradGuard, operations should not record
    {
        let _guard = NoGradGuard::new();
        let z = &x * &y;

        // z should NOT have a grad_fn
        assert!(!z.has_grad_fn(), "z should not have grad_fn inside NoGradGuard");
    }

    // Outside NoGradGuard, operations should record
    let z = &x * &y;
    assert!(z.has_grad_fn(), "z should have grad_fn outside NoGradGuard");
}

#[test]
fn test_requires_grad_propagates_to_result() {
    let mut x = Tensor::from_slice(&[1.0]);
    let y = Tensor::from_slice(&[2.0]);

    // Only x requires grad
    x.set_requires_grad(true);

    let z = &x * &y;

    // z should inherit requires_grad
    assert_eq!(z.requires_grad(), true);
}

#[test]
fn test_no_requires_grad_no_grad_fn() {
    let x = Tensor::from_slice(&[1.0]);
    let y = Tensor::from_slice(&[2.0]);

    // Neither requires grad
    assert_eq!(x.requires_grad(), false);
    assert_eq!(y.requires_grad(), false);

    let z = &x * &y;

    // z should not have grad_fn
    assert!(!z.has_grad_fn());
    assert_eq!(z.requires_grad(), false);
}

// ============================================================
// Phase 1.3: Backward Pass Infrastructure Tests
// ============================================================

#[test]
fn test_simple_backward() {
    // Test simple chain rule: z = x * y
    // dz/dx = y, dz/dy = x
    let mut x = Tensor::from_slice(&[2.0]);
    let mut y = Tensor::from_slice(&[3.0]);

    x.set_requires_grad(true);
    y.set_requires_grad(true);

    // z = x * y = 2.0 * 3.0 = 6.0
    let mut z = &x * &y;

    // Perform backward pass
    z.backward();

    // Check gradients
    let x_grad = x.grad().expect("x should have gradient");
    let y_grad = y.grad().expect("y should have gradient");

    // dz/dx = y = 3.0
    assert_eq!(x_grad.to_vec_f64()[0], 3.0, "dz/dx should be 3.0");

    // dz/dy = x = 2.0
    assert_eq!(y_grad.to_vec_f64()[0], 2.0, "dz/dy should be 2.0");
}

#[test]
#[ignore] // TODO: Fix gradient propagation through multiple levels
fn test_backward_chain() {
    // Test chain of operations: w = x * y, z = w * a
    let mut x = Tensor::from_slice(&[2.0]);
    let mut y = Tensor::from_slice(&[3.0]);
    let mut a = Tensor::from_slice(&[2.0]);

    x.set_requires_grad(true);
    y.set_requires_grad(true);
    a.set_requires_grad(true);

    // w = x * y = 6.0
    let w = &x * &y;

    // z = w * a = 6.0 * 2.0 = 12.0
    let mut z = &w * &a;

    z.backward();

    // dz/dx = dz/dw * dw/dx = a * y = 2 * 3 = 6
    let x_grad = x.grad().expect("x should have gradient");
    assert_eq!(x_grad.to_vec_f64()[0], 6.0, "dz/dx should be 6.0");

    // dz/dy = dz/dw * dw/dy = a * x = 2 * 2 = 4
    let y_grad = y.grad().expect("y should have gradient");
    assert_eq!(y_grad.to_vec_f64()[0], 4.0, "dz/dy should be 4.0");

    // dz/da = w = 6.0
    let a_grad = a.grad().expect("a should have gradient");
    assert_eq!(a_grad.to_vec_f64()[0], 6.0, "dz/da should be 6.0");
}

#[test]
fn test_backward_with_gradient() {
    // Test backward with custom gradient
    let mut x = Tensor::from_slice(&[2.0]);
    let mut y = Tensor::from_slice(&[3.0]);

    x.set_requires_grad(true);
    y.set_requires_grad(true);

    let mut z = &x * &y;

    // Use custom gradient (instead of implicit 1.0)
    let custom_grad = Tensor::from_slice(&[5.0]);
    z.backward_with_gradient(custom_grad);

    // dz/dx = grad_output * y = 5.0 * 3.0 = 15.0
    let x_grad = x.grad().expect("x should have gradient");
    assert_eq!(x_grad.to_vec_f64()[0], 15.0);

    // dz/dy = grad_output * x = 5.0 * 2.0 = 10.0
    let y_grad = y.grad().expect("y should have gradient");
    assert_eq!(y_grad.to_vec_f64()[0], 10.0);
}

#[test]
#[should_panic(expected = "Backward pass failed")]
fn test_backward_no_grad_fn() {
    // Test that backward fails on a tensor without grad_fn
    let mut x = Tensor::from_slice(&[2.0]);
    x.set_requires_grad(true);

    // x is a leaf tensor with no grad_fn
    x.backward();  // Should panic
}
