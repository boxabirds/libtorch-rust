use crate::{Device, Kind, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Variable store for managing model parameters
#[derive(Clone)]
pub struct VarStore {
    variables: Arc<Mutex<HashMap<String, Tensor>>>,
    device: Device,
}

impl VarStore {
    /// Create a new variable store
    pub fn new(device: Device) -> Self {
        VarStore {
            variables: Arc::new(Mutex::new(HashMap::new())),
            device,
        }
    }

    /// Get the root path
    pub fn root(&self) -> Path {
        Path {
            path: String::new(),
            var_store: self.clone(),
        }
    }

    /// Get all trainable variables
    pub fn trainable_variables(&self) -> Vec<Tensor> {
        let vars = self.variables.lock().unwrap();
        vars.values().cloned().collect()
    }

    /// Freeze all variables (disable gradient tracking)
    pub fn freeze(&mut self) {
        // TODO: Implement gradient freezing
    }

    /// Unfreeze all variables (enable gradient tracking)
    pub fn unfreeze(&mut self) {
        // TODO: Implement gradient unfreezing
    }

    /// Save the variable store to a file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), std::io::Error> {
        // TODO: Implement serialization
        let _ = path;
        Ok(())
    }

    /// Load the variable store from a file
    pub fn load<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<(), std::io::Error> {
        // TODO: Implement deserialization
        let _ = path;
        Ok(())
    }
}

/// Path for hierarchical parameter naming
#[derive(Clone)]
pub struct Path {
    path: String,
    var_store: VarStore,
}

impl Path {
    /// Create a sub-path
    pub fn sub(&self, name: &str) -> Path {
        let new_path = if self.path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.path, name)
        };

        Path {
            path: new_path,
            var_store: self.var_store.clone(),
        }
    }

    /// Create a variable with the given shape and initialization
    pub fn var(&self, name: &str, shape: &[i64], init: Init) -> Tensor {
        let full_name = if self.path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.path, name)
        };

        let mut vars = self.var_store.variables.lock().unwrap();

        if let Some(tensor) = vars.get(&full_name) {
            return tensor.clone();
        }

        let tensor = init.create_tensor(shape, self.var_store.device);
        vars.insert(full_name, tensor.clone());
        tensor
    }

    /// Create a zeros tensor
    pub fn zeros(&self, name: &str, shape: &[i64]) -> Tensor {
        self.var(name, shape, Init::Const(0.0))
    }

    /// Create an ones tensor
    pub fn ones(&self, name: &str, shape: &[i64]) -> Tensor {
        self.var(name, shape, Init::Const(1.0))
    }

    /// Get the device
    pub fn device(&self) -> Device {
        self.var_store.device
    }
}

/// Initialization methods for parameters
#[derive(Clone, Copy)]
pub enum Init {
    Const(f64),
    Uniform { lo: f64, up: f64 },
    Randn { mean: f64, stdev: f64 },
    KaimingUniform,
}

impl Init {
    fn create_tensor(&self, shape: &[i64], device: Device) -> Tensor {
        match self {
            Init::Const(val) => {
                let mut tensor = Tensor::zeros(shape, Kind::Float, device);
                tensor.fill_(*val);
                tensor
            }
            Init::Uniform { lo, up } => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let numel: usize = shape.iter().map(|&x| x as usize).product();
                let data: Vec<f32> = (0..numel)
                    .map(|_| rng.gen_range(*lo as f32..*up as f32))
                    .collect();
                Tensor::from_slice(&data).reshape(shape)
            }
            Init::Randn { mean, stdev } => {
                use rand_distr::{Distribution, Normal};
                let normal = Normal::new(*mean, *stdev).unwrap();
                let mut rng = rand::thread_rng();
                let numel: usize = shape.iter().map(|&x| x as usize).product();
                let data: Vec<f32> = (0..numel).map(|_| normal.sample(&mut rng) as f32).collect();
                Tensor::from_slice(&data).reshape(shape)
            }
            Init::KaimingUniform => {
                // Simplified Kaiming initialization
                let fan_in = if shape.len() >= 2 { shape[1] as f64 } else { 1.0 };
                let bound = (3.0 / fan_in).sqrt();
                Init::Uniform { lo: -bound, up: bound }.create_tensor(shape, device)
            }
        }
    }
}
