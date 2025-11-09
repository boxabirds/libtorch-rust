use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tch::{Device, Kind, Tensor};

/// Benchmark tensor creation operations
fn bench_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    // Benchmark different tensor sizes
    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("zeros", size), size, |b, &size| {
            b.iter(|| {
                black_box(Tensor::zeros(&[size as i64], Kind::Float, Device::Cpu))
            });
        });

        group.bench_with_input(BenchmarkId::new("ones", size), size, |b, &size| {
            b.iter(|| {
                black_box(Tensor::ones(&[size as i64], Kind::Float, Device::Cpu))
            });
        });

        group.bench_with_input(BenchmarkId::new("from_slice", size), size, |b, &size| {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            b.iter(|| {
                black_box(Tensor::from_slice(&data))
            });
        });
    }

    group.finish();
}

/// Benchmark 2D tensor creation
fn bench_tensor_2d_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_2d_creation");

    for dim in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("zeros_2d", dim), dim, |b, &dim| {
            b.iter(|| {
                black_box(Tensor::zeros(&[dim as i64, dim as i64], Kind::Float, Device::Cpu))
            });
        });
    }

    group.finish();
}

/// Benchmark element-wise operations
fn bench_elementwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_operations");

    for size in [100, 1000, 10000].iter() {
        let a = Tensor::ones(&[*size as i64], Kind::Float, Device::Cpu);
        let b = Tensor::ones(&[*size as i64], Kind::Float, Device::Cpu);

        group.bench_with_input(BenchmarkId::new("add", size), size, |bench, _| {
            bench.iter(|| {
                black_box(&a + &b)
            });
        });

        group.bench_with_input(BenchmarkId::new("sub", size), size, |bench, _| {
            bench.iter(|| {
                black_box(&a - &b)
            });
        });

        group.bench_with_input(BenchmarkId::new("mul", size), size, |bench, _| {
            bench.iter(|| {
                black_box(&a * &b)
            });
        });

        group.bench_with_input(BenchmarkId::new("div", size), size, |bench, _| {
            bench.iter(|| {
                black_box(&a / &b)
            });
        });
    }

    group.finish();
}

/// Benchmark scalar operations
fn bench_scalar_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_operations");

    for size in [100, 1000, 10000].iter() {
        let a = Tensor::ones(&[*size as i64], Kind::Float, Device::Cpu);

        group.bench_with_input(BenchmarkId::new("add_scalar", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.add_scalar(2.5))
            });
        });

        group.bench_with_input(BenchmarkId::new("mul_scalar", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.mul_scalar(2.5))
            });
        });
    }

    group.finish();
}

/// Benchmark matrix multiplication
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");

    // Test different matrix sizes
    for dim in [10, 50, 100, 200, 500].iter() {
        let a = Tensor::ones(&[*dim as i64, *dim as i64], Kind::Float, Device::Cpu);
        let b = Tensor::ones(&[*dim as i64, *dim as i64], Kind::Float, Device::Cpu);

        group.bench_with_input(BenchmarkId::new("matmul", dim), dim, |bench, _| {
            bench.iter(|| {
                black_box(a.matmul(&b))
            });
        });
    }

    group.finish();
}

/// Benchmark activation functions
fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_functions");

    for size in [100, 1000, 10000].iter() {
        let a = Tensor::ones(&[*size as i64], Kind::Float, Device::Cpu);

        group.bench_with_input(BenchmarkId::new("relu", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.relu())
            });
        });

        group.bench_with_input(BenchmarkId::new("sigmoid", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.sigmoid())
            });
        });
    }

    // 2D softmax
    for dim in [10, 50, 100].iter() {
        let a = Tensor::ones(&[*dim as i64, *dim as i64], Kind::Float, Device::Cpu);

        group.bench_with_input(BenchmarkId::new("softmax", dim), dim, |bench, _| {
            bench.iter(|| {
                black_box(a.softmax(1, Kind::Float))
            });
        });
    }

    group.finish();
}

/// Benchmark reduction operations
fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");

    for size in [100, 1000, 10000, 100000].iter() {
        let a = Tensor::ones(&[*size as i64], Kind::Float, Device::Cpu);

        group.bench_with_input(BenchmarkId::new("sum", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.sum(Kind::Float))
            });
        });

        group.bench_with_input(BenchmarkId::new("mean", size), size, |bench, _| {
            bench.iter(|| {
                black_box(a.mean(Kind::Float))
            });
        });
    }

    group.finish();
}

/// Benchmark shape operations
fn bench_shape_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_operations");

    let a = Tensor::ones(&[100, 100], Kind::Float, Device::Cpu);

    group.bench_function("reshape", |b| {
        b.iter(|| {
            black_box(a.reshape(&[10000]))
        });
    });

    group.bench_function("transpose", |b| {
        b.iter(|| {
            black_box(a.transpose(0, 1))
        });
    });

    group.bench_function("unsqueeze", |b| {
        b.iter(|| {
            black_box(a.unsqueeze(0))
        });
    });

    group.bench_function("squeeze", |b| {
        let a_unsqueezed = a.unsqueeze(0);
        b.iter(|| {
            black_box(a_unsqueezed.squeeze())
        });
    });

    group.finish();
}

/// Benchmark broadcasting operations
fn bench_broadcasting(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcasting");

    for dim in [10, 50, 100].iter() {
        let a = Tensor::ones(&[*dim as i64, *dim as i64], Kind::Float, Device::Cpu);
        let b = Tensor::ones(&[*dim as i64], Kind::Float, Device::Cpu);

        group.bench_with_input(BenchmarkId::new("broadcast_2d_1d", dim), dim, |bench, _| {
            bench.iter(|| {
                black_box(&a + &b)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_creation,
    bench_tensor_2d_creation,
    bench_elementwise_ops,
    bench_scalar_ops,
    bench_matmul,
    bench_activations,
    bench_reductions,
    bench_shape_ops,
    bench_broadcasting,
);
criterion_main!(benches);
