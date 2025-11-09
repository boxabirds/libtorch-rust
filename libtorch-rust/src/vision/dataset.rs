use crate::Tensor;

/// Trait for datasets
pub trait Dataset {
    /// Get the size of the dataset
    fn len(&self) -> usize;

    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an item from the dataset
    fn get(&self, index: usize) -> Option<(Tensor, Tensor)>;
}

/// Iterator for datasets
pub struct DatasetIter<D: Dataset> {
    dataset: D,
    index: usize,
}

impl<D: Dataset> Iterator for DatasetIter<D> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.len() {
            return None;
        }

        let item = self.dataset.get(self.index);
        self.index += 1;
        item
    }
}
