use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tch::nn::{linear, Linear, Module, VarStore};
use tch::{Device, Kind, Tensor};

/// Benchmark VarStore operations
fn bench_varstore(c: &mut Criterion) {
    let mut group = c.benchmark_group("varstore");

    group.bench_function("create_varstore", |b| {
        b.iter(|| {
            black_box(VarStore::new(Device::Cpu))
        });
    });

    group.bench_function("varstore_with_variables", |b| {
        b.iter(|| {
            let vs = VarStore::new(Device::Cpu);
            let root = vs.root();
            black_box(root.var("w", &[100, 100], Kind::Float));
            black_box(root.var("b", &[100], Kind::Float));
        });
    });

    group.finish();
}

/// Benchmark Linear layer operations
fn bench_linear_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_layer");

    // Test different layer sizes
    for (in_dim, out_dim) in [(10, 10), (100, 100), (784, 128), (128, 10)].iter() {
        let vs = VarStore::new(Device::Cpu);
        let layer = linear(&vs.root(), *in_dim, *out_dim);

        group.bench_with_input(
            BenchmarkId::new("forward_single", format!("{}x{}", in_dim, out_dim)),
            &(*in_dim, *out_dim),
            |b, &(in_d, _)| {
                let input = Tensor::ones(&[in_d as i64], Kind::Float, Device::Cpu);
                b.iter(|| {
                    black_box(layer.forward(&input))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("forward_batch", format!("{}x{}", in_dim, out_dim)),
            &(*in_dim, *out_dim),
            |b, &(in_d, _)| {
                let input = Tensor::ones(&[32, in_d as i64], Kind::Float, Device::Cpu);
                b.iter(|| {
                    black_box(layer.forward(&input))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multi-layer neural networks
fn bench_multilayer_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("multilayer_network");

    // Two-layer network (like in examples)
    let vs = VarStore::new(Device::Cpu);
    let layer1 = linear(&vs.root().sub("layer1"), 784, 128);
    let layer2 = linear(&vs.root().sub("layer2"), 128, 10);

    group.bench_function("two_layer_forward", |b| {
        let input = Tensor::ones(&[32, 784], Kind::Float, Device::Cpu);
        b.iter(|| {
            let hidden = layer1.forward(&input).relu();
            black_box(layer2.forward(&hidden))
        });
    });

    // Three-layer network
    let vs3 = VarStore::new(Device::Cpu);
    let l1 = linear(&vs3.root().sub("layer1"), 784, 256);
    let l2 = linear(&vs3.root().sub("layer2"), 256, 128);
    let l3 = linear(&vs3.root().sub("layer3"), 128, 10);

    group.bench_function("three_layer_forward", |b| {
        let input = Tensor::ones(&[32, 784], Kind::Float, Device::Cpu);
        b.iter(|| {
            let h1 = l1.forward(&input).relu();
            let h2 = l2.forward(&h1).relu();
            black_box(l3.forward(&h2))
        });
    });

    group.finish();
}

/// Benchmark complete forward pass with activations
fn bench_forward_with_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_with_activations");

    let vs = VarStore::new(Device::Cpu);
    let layer = linear(&vs.root(), 100, 100);

    group.bench_function("linear_relu", |b| {
        let input = Tensor::ones(&[32, 100], Kind::Float, Device::Cpu);
        b.iter(|| {
            black_box(layer.forward(&input).relu())
        });
    });

    group.bench_function("linear_sigmoid", |b| {
        let input = Tensor::ones(&[32, 100], Kind::Float, Device::Cpu);
        b.iter(|| {
            black_box(layer.forward(&input).sigmoid())
        });
    });

    group.bench_function("linear_softmax", |b| {
        let input = Tensor::ones(&[32, 100], Kind::Float, Device::Cpu);
        b.iter(|| {
            black_box(layer.forward(&input).softmax(1, Kind::Float))
        });
    });

    group.finish();
}

/// Benchmark MNIST-like network
fn bench_mnist_like_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("mnist_network");

    // Typical MNIST architecture: 784 -> 128 -> 64 -> 10
    let vs = VarStore::new(Device::Cpu);
    let fc1 = linear(&vs.root().sub("fc1"), 784, 128);
    let fc2 = linear(&vs.root().sub("fc2"), 128, 64);
    let fc3 = linear(&vs.root().sub("fc3"), 64, 10);

    group.bench_function("mnist_forward_pass", |b| {
        let input = Tensor::ones(&[1, 784], Kind::Float, Device::Cpu);
        b.iter(|| {
            let x = fc1.forward(&input).relu();
            let x = fc2.forward(&x).relu();
            black_box(fc3.forward(&x))
        });
    });

    group.bench_function("mnist_forward_batch_32", |b| {
        let input = Tensor::ones(&[32, 784], Kind::Float, Device::Cpu);
        b.iter(|| {
            let x = fc1.forward(&input).relu();
            let x = fc2.forward(&x).relu();
            black_box(fc3.forward(&x))
        });
    });

    group.bench_function("mnist_forward_batch_128", |b| {
        let input = Tensor::ones(&[128, 784], Kind::Float, Device::Cpu);
        b.iter(|| {
            let x = fc1.forward(&input).relu();
            let x = fc2.forward(&x).relu();
            black_box(fc3.forward(&x))
        });
    });

    group.finish();
}

/// Benchmark deeper networks
fn bench_deep_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("deep_network");

    // 5-layer network
    let vs = VarStore::new(Device::Cpu);
    let layers: Vec<Linear> = (0..5)
        .map(|i| linear(&vs.root().sub(&format!("layer{}", i)), 256, 256))
        .collect();

    group.bench_function("five_layer_forward", |b| {
        let input = Tensor::ones(&[32, 256], Kind::Float, Device::Cpu);
        b.iter(|| {
            let mut x = input.shallow_clone();
            for layer in &layers {
                x = layer.forward(&x).relu();
            }
            black_box(x)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_varstore,
    bench_linear_layer,
    bench_multilayer_network,
    bench_forward_with_activations,
    bench_mnist_like_network,
    bench_deep_network,
);
criterion_main!(benches);
