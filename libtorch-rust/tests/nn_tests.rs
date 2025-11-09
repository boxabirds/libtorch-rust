// Tests ported from PyTorch's test/cpp/api/modules.cpp and module.cpp

use tch::nn::{linear, Module, VarStore};
use tch::{Device, Kind};

#[test]
fn test_var_store_creation() {
    let vs = VarStore::new(Device::Cpu);
    let root = vs.root();

    let vars = vs.trainable_variables();
    assert_eq!(vars.len(), 0);
}

#[test]
fn test_var_store_variables() {
    let vs = VarStore::new(Device::Cpu);
    let root = vs.root();

    let _v1 = root.zeros("weight", &[2, 3]);
    let _v2 = root.ones("bias", &[3]);

    let vars = vs.trainable_variables();
    assert_eq!(vars.len(), 2);
}

#[test]
fn test_linear_layer_creation() {
    let vs = VarStore::new(Device::Cpu);
    let root = vs.root();

    let lin = linear(&root, 10, 5);

    // Check weight shape: [out_dim, in_dim]
    assert_eq!(lin.ws.size(), vec![5, 10]);

    // Check bias shape: [out_dim]
    assert!(lin.bs.is_some());
    let bias = lin.bs.as_ref().unwrap();
    assert_eq!(bias.size(), vec![5]);
}

#[test]
fn test_linear_forward() {
    let vs = VarStore::new(Device::Cpu);
    let root = vs.root();

    let lin = linear(&root, 3, 2);

    // Create input: batch_size=1, features=3
    let input = tch::Tensor::ones(&[1, 3], Kind::Float, Device::Cpu);

    let output = lin.forward(&input);

    // Output should be: batch_size=1, out_features=2
    assert_eq!(output.size(), vec![1, 2]);
}

#[test]
fn test_linear_forward_batch() {
    let vs = VarStore::new(Device::Cpu);
    let root = vs.root();

    let lin = linear(&root, 3, 2);

    // Create input: batch_size=4, features=3
    let input = tch::Tensor::ones(&[4, 3], Kind::Float, Device::Cpu);

    let output = lin.forward(&input);

    // Output should be: batch_size=4, out_features=2
    assert_eq!(output.size(), vec![4, 2]);
}

#[test]
fn test_path_sub() {
    let vs = VarStore::new(Device::Cpu);
    let root = vs.root();

    let sub1 = root.sub("layer1");
    let _v1 = sub1.zeros("weight", &[2, 3]);

    let sub2 = root.sub("layer2");
    let _v2 = sub2.ones("weight", &[3, 4]);

    let vars = vs.trainable_variables();
    assert_eq!(vars.len(), 2);
}

#[test]
fn test_sequential() {
    use tch::nn::Sequential;

    let seq = Sequential::new()
        .add_fn(|x| x.relu())
        .add_fn(|x| x.mul_scalar(2.0));

    let input = tch::Tensor::of_slice(&[-1.0, 0.0, 1.0, 2.0]);
    let output = seq.forward(&input);

    let result = output.to_vec_f64();
    // After relu: [0, 0, 1, 2]
    // After *2: [0, 0, 2, 4]
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 2.0);
    assert_eq!(result[3], 4.0);
}

#[test]
fn test_optimizer_creation() {
    use tch::nn::{OptimizerConfig, Sgd};

    let vs = VarStore::new(Device::Cpu);
    let _opt = Sgd::new(0.01).build(&vs);
}

#[test]
fn test_adam_optimizer_creation() {
    use tch::nn::{Adam, OptimizerConfig};

    let vs = VarStore::new(Device::Cpu);
    let _opt = Adam::new(0.001).build(&vs);
}
