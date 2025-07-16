import os
import requests
import numpy as np
import pandas as pd
import torch
from typing import Iterator, Tuple, Optional
from tqdm import tqdm
import urllib.request
import zipfile
import gzip
import struct


def download_file(url: str, filepath: str, description: str = None) -> None:
    """Download a file from URL to filepath with progress bar."""
    if os.path.exists(filepath):
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with tqdm(unit='B', unit_scale=True, desc=description or "Downloading") as pbar:
        def update_progress(block_num, block_size, total_size):
            if total_size > 0:
                pbar.total = total_size
            pbar.update(block_size)
        
        urllib.request.urlretrieve(url, filepath, reporthook=update_progress)


def download_rotating_mnist(data_dir: str = "data/rotating_mnist") -> str:
    """Download Rotating MNIST dataset."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the rotating MNIST data
    url = "https://github.com/google-research/rotating-mnist/raw/main/sequence_test.npy"
    filepath = os.path.join(data_dir, "sequence_test.npy")
    
    try:
        download_file(url, filepath, "Rotating MNIST")
    except Exception as e:
        print(f"Error downloading from Google Research repo: {e}")
        # Create synthetic rotating MNIST data if download fails
        print("Creating synthetic rotating MNIST data...")
        create_synthetic_rotating_mnist(data_dir)
    
    return data_dir


def create_synthetic_rotating_mnist(data_dir: str) -> None:
    """Create synthetic rotating MNIST data for testing."""
    # Create a more challenging synthetic dataset
    np.random.seed(42)
    
    # Generate 60000 samples of 784 features (28x28 images)
    n_samples = 60000
    n_features = 784
    
    # Create base patterns (10 classes) with overlapping features
    base_patterns = []
    for i in range(10):
        # Start with random noise for each class
        pattern = np.random.randn(28, 28) * 0.5
        
        # Add some class-specific structure, but make them similar
        center = (14, 14)
        
        # All classes have some common structure to make it challenging
        for y in range(28):
            for x in range(28):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if dist < 12:  # All classes have a central region
                    pattern[y, x] += 0.5
        
        # Add slight class-specific variations
        if i % 2 == 0:  # Even classes
            pattern[10:18, 10:18] += 0.3
        else:  # Odd classes
            pattern[8:20, 8:20] += 0.2
        
        # Add very subtle class-specific features
        pattern[i+5:i+15, i+5:i+15] += 0.1
        
        base_patterns.append(pattern)
    
    # Generate sequence with realistic difficulty
    data = []
    labels = []
    
    for i in range(n_samples):
        # Random labels with some structure
        class_label = np.random.randint(0, 10)
        
        # Start with base pattern
        pattern = base_patterns[class_label].copy()
        
        # Add significant noise to make it challenging
        pattern += np.random.randn(28, 28) * 0.8
        
        # Add rotation-like drift over time
        if i > 0:
            angle = (i / 1000) * 5  # 5 degrees per 1000 samples
            rotation_factor = np.sin(np.radians(angle)) * 0.3
            
            # Apply rotation-like transformation
            for y in range(28):
                for x in range(28):
                    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    if dist > 0:
                        pattern[y, x] += rotation_factor * np.sin(dist / 3.0)
        
        data.append(pattern.flatten())
        labels.append(class_label)
    
    # Save as numpy arrays
    np.save(os.path.join(data_dir, "sequence_test.npy"), np.array(data))
    np.save(os.path.join(data_dir, "labels.npy"), np.array(labels))
    
    print(f"Created synthetic rotating MNIST with {n_samples} samples")


def download_covtype(data_dir: str = "data/covtype") -> str:
    """Download COVTYPE dataset."""
    os.makedirs(data_dir, exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    filepath = os.path.join(data_dir, "covtype.data.gz")
    
    try:
        download_file(url, filepath, "COVTYPE")
        
        # Extract and convert to numpy
        data_path = os.path.join(data_dir, "covtype.data")
        if not os.path.exists(data_path):
            with gzip.open(filepath, 'rb') as f_in:
                with open(data_path, 'wb') as f_out:
                    f_out.write(f_in.read())
    except Exception as e:
        print(f"Error downloading COVTYPE: {e}")
        print("Creating synthetic COVTYPE data...")
        create_synthetic_covtype(data_dir)
    
    return data_dir


def create_synthetic_covtype(data_dir: str) -> None:
    """Create synthetic COVTYPE-like data for testing."""
    np.random.seed(42)
    
    # COVTYPE has 54 features and 7 classes
    n_samples = 100000
    n_features = 54
    n_classes = 7
    
    # Create synthetic data with some structure
    data = []
    labels = []
    
    # Create class centroids in feature space
    centroids = np.random.randn(n_classes, n_features) * 2
    
    for i in range(n_samples):
        # Choose class randomly
        class_label = np.random.randint(0, n_classes)
        
        # Generate sample around class centroid with noise
        sample = centroids[class_label] + np.random.randn(n_features) * 0.8
        
        # Add some structure to make it more realistic
        # Some features are categorical (0/1)
        sample[:10] = (sample[:10] > 0).astype(float)
        
        # Some features are counts (non-negative integers)
        sample[10:20] = np.maximum(0, np.round(sample[10:20]))
        
        # Some features are continuous
        sample[20:] = sample[20:]
        
        data.append(sample)
        labels.append(class_label)
    
    # Create CSV-like data
    full_data = np.column_stack([data, labels])
    
    # Save to CSV format
    data_path = os.path.join(data_dir, "covtype.data")
    np.savetxt(data_path, full_data, delimiter=',', fmt='%g')
    
    print(f"Created synthetic COVTYPE with {n_samples} samples")


def load_rotating_mnist(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load rotating MNIST data."""
    data_path = os.path.join(data_dir, "sequence_test.npy")
    labels_path = os.path.join(data_dir, "labels.npy")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    
    data = np.load(data_path)
    
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
    else:
        # Generate labels if not available
        labels = np.arange(len(data)) % 10
    
    return data, labels


def load_covtype(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load COVTYPE dataset."""
    data_path = os.path.join(data_dir, "covtype.data")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    
    # Load covtype data
    data = pd.read_csv(data_path, header=None)
    
    # Separate features and labels
    X = data.iloc[:, :-1].values  # All columns except last
    y = data.iloc[:, -1].values - 1  # Last column, convert to 0-indexed
    
    return X, y


class DataStreamGenerator:
    """Base class for data stream generators."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.rng = np.random.RandomState(seed)
    
    def stream(self, T: int) -> Iterator[Tuple[np.ndarray, int]]:
        """Generate a stream of T samples."""
        raise NotImplementedError


class IIDStreamGenerator(DataStreamGenerator):
    """IID shuffle stream generator."""
    
    def stream(self, T: int) -> Iterator[Tuple[np.ndarray, int]]:
        """Generate IID shuffled stream."""
        for t in range(T):
            idx = self.rng.randint(0, self.n_samples)
            yield self.X[idx], self.y[idx]


class GradualDriftStreamGenerator(DataStreamGenerator):
    """Gradual drift stream generator."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        super().__init__(X, y, seed)
        self.rotation_angle = 0.0
        self.feature_drift = np.zeros(X.shape[1])
    
    def stream(self, T: int) -> Iterator[Tuple[np.ndarray, int]]:
        """Generate gradual drift stream."""
        for t in range(T):
            # Update drift every 1000 steps
            if t % 1000 == 0:
                self.rotation_angle += 5.0  # +5 degrees for rotation
                # Feature drift for COVTYPE
                self.feature_drift += self.rng.normal(0, 0.01, self.X.shape[1])
            
            # Get sample
            idx = self.rng.randint(0, self.n_samples)
            x, y = self.X[idx].copy(), self.y[idx]
            
            # Apply drift
            if self.X.shape[1] == 784:  # Rotating MNIST
                x = self._apply_rotation_drift(x)
            else:  # COVTYPE
                x = self._apply_feature_drift(x)
            
            yield x, y
    
    def _apply_rotation_drift(self, x: np.ndarray) -> np.ndarray:
        """Apply rotation drift to image data."""
        # Simple rotation simulation by adding structured noise
        angle_rad = np.radians(self.rotation_angle)
        rotation_factor = np.sin(angle_rad) * 0.1
        
        # Reshape to 28x28 for rotation effect
        img = x.reshape(28, 28)
        
        # Apply rotation-like transformation
        center = (14, 14)
        for i in range(28):
            for j in range(28):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                if dist > 0:
                    img[i, j] += rotation_factor * (dist / 20.0) * self.rng.normal(0, 0.1)
        
        return img.flatten()
    
    def _apply_feature_drift(self, x: np.ndarray) -> np.ndarray:
        """Apply feature drift to tabular data."""
        return x + self.feature_drift


class AdversarialStreamGenerator(DataStreamGenerator):
    """Adversarial permute stream generator."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        super().__init__(X, y, seed)
        self.permutation = np.arange(self.X.shape[1])
    
    def stream(self, T: int) -> Iterator[Tuple[np.ndarray, int]]:
        """Generate adversarial permutation stream."""
        for t in range(T):
            # Update permutation every 500 steps
            if t % 500 == 0:
                # Random swap
                i, j = self.rng.choice(len(self.permutation), 2, replace=False)
                self.permutation[i], self.permutation[j] = self.permutation[j], self.permutation[i]
            
            # Get sample and apply permutation
            idx = self.rng.randint(0, self.n_samples)
            x, y = self.X[idx].copy(), self.y[idx]
            x = x[self.permutation]
            
            yield x, y


def get_dataset_generator(dataset: str, stream_type: str, seed: int = 42) -> DataStreamGenerator:
    """Get the appropriate dataset and stream generator."""
    if dataset == "rotmnist":
        data_dir = download_rotating_mnist()
        X, y = load_rotating_mnist(data_dir)
    elif dataset == "covtype":
        data_dir = download_covtype()
        X, y = load_covtype(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if stream_type == "iid":
        return IIDStreamGenerator(X, y, seed)
    elif stream_type == "drift":
        return GradualDriftStreamGenerator(X, y, seed)
    elif stream_type == "adv":
        return AdversarialStreamGenerator(X, y, seed)
    else:
        raise ValueError(f"Unknown stream type: {stream_type}")