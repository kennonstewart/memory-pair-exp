import numpy as np
from typing import Optional, List
from abc import ABC, abstractmethod
import torch
from .memory_pair import StreamNewtonMemoryPair


class OnlineAlgorithm(ABC):
    """Base class for online learning algorithms."""
    
    def __init__(self, n_features: int, n_classes: int, seed: int = 42):
        self.n_features = n_features
        self.n_classes = n_classes
        self.rng = np.random.RandomState(seed)
        self.t = 0
        self.weights = None
        self.cumulative_regret = 0.0
        self.regret_history = []
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> int:
        """Make a prediction for input x."""
        pass
    
    @abstractmethod
    def update(self, x: np.ndarray, y: int, loss: float) -> None:
        """Update the algorithm with new sample and loss."""
        pass
    
    def step(self, x: np.ndarray, y: int) -> float:
        """Perform one step of online learning."""
        # Make prediction
        pred = self.predict(x)
        
        # Calculate loss (0-1 loss for classification)
        loss = 1.0 if pred != y else 0.0
        
        # Update cumulative regret
        self.cumulative_regret += loss
        self.regret_history.append(self.cumulative_regret)
        
        # Update algorithm
        self.update(x, y, loss)
        
        self.t += 1
        return loss


class OnlineSGD(OnlineAlgorithm):
    """Online Stochastic Gradient Descent."""
    
    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.01, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.learning_rate = learning_rate
        self.weights = np.zeros((n_classes, n_features))  # Initialize to zeros
    
    def predict(self, x: np.ndarray) -> int:
        """Predict using current weights."""
        scores = np.dot(self.weights, x)
        return np.argmax(scores)
    
    def update(self, x: np.ndarray, y: int, loss: float) -> None:
        """Update weights using SGD."""
        if loss > 0:  # Only update if we made a mistake
            # Get current prediction
            scores = np.dot(self.weights, x)
            pred = np.argmax(scores)
            
            # Update weights (perceptron-like update)
            self.weights[y] += self.learning_rate * x
            self.weights[pred] -= self.learning_rate * x


class AdaGrad(OnlineAlgorithm):
    """AdaGrad algorithm."""
    
    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.01, epsilon: float = 1e-8, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.weights = np.zeros((n_classes, n_features))  # Initialize to zeros
        self.G = np.zeros((n_classes, n_features))  # Accumulated squared gradients
    
    def predict(self, x: np.ndarray) -> int:
        """Predict using current weights."""
        scores = np.dot(self.weights, x)
        return np.argmax(scores)
    
    def update(self, x: np.ndarray, y: int, loss: float) -> None:
        """Update weights using AdaGrad."""
        if loss > 0:  # Only update if we made a mistake
            # Get current prediction
            scores = np.dot(self.weights, x)
            pred = np.argmax(scores)
            
            # Calculate gradients
            grad_correct = -x  # Gradient for correct class
            grad_pred = x      # Gradient for predicted class
            
            # Update accumulated gradients
            self.G[y] += grad_correct ** 2
            self.G[pred] += grad_pred ** 2
            
            # AdaGrad update
            self.weights[y] -= self.learning_rate * grad_correct / (np.sqrt(self.G[y]) + self.epsilon)
            self.weights[pred] -= self.learning_rate * grad_pred / (np.sqrt(self.G[pred]) + self.epsilon)


class OnlineNewtonStep(OnlineAlgorithm):
    """Online Newton Step algorithm."""
    
    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.01, regularization: float = 0.01, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.weights = np.zeros((n_classes, n_features))  # Initialize to zeros
        self.A = np.eye(n_features) * regularization  # Hessian approximation
    
    def predict(self, x: np.ndarray) -> int:
        """Predict using current weights."""
        scores = np.dot(self.weights, x)
        return np.argmax(scores)
    
    def update(self, x: np.ndarray, y: int, loss: float) -> None:
        """Update weights using Online Newton Step."""
        if loss > 0:  # Only update if we made a mistake
            # Get current prediction
            scores = np.dot(self.weights, x)
            pred = np.argmax(scores)
            
            # Update Hessian approximation
            self.A += np.outer(x, x)
            
            # Calculate Newton step
            try:
                A_inv = np.linalg.inv(self.A)
                
                # Calculate gradients
                grad_correct = -x
                grad_pred = x
                
                # Newton update
                self.weights[y] -= self.learning_rate * np.dot(A_inv, grad_correct)
                self.weights[pred] -= self.learning_rate * np.dot(A_inv, grad_pred)
                
            except np.linalg.LinAlgError:
                # Fall back to SGD if matrix is singular
                self.weights[y] += self.learning_rate * x
                self.weights[pred] -= self.learning_rate * x


class MemoryPairOnlineLBFGS(OnlineAlgorithm):
    """Memory-Pair Online L-BFGS algorithm."""
    
    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.001, 
                 memory_size: int = 5, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.weights = np.zeros((n_classes, n_features))  # Initialize to zeros
        
        # L-BFGS memory
        self.s_memory = []  # Step differences
        self.y_memory = []  # Gradient differences
        self.rho_memory = []  # 1 / (y^T s)
        
        self.prev_grad = None
        self.prev_weights = None
    
    def predict(self, x: np.ndarray) -> int:
        """Predict using current weights."""
        scores = np.dot(self.weights, x)
        return np.argmax(scores)
    
    def update(self, x: np.ndarray, y: int, loss: float) -> None:
        """Update weights using Memory-Pair L-BFGS."""
        if loss > 0:  # Only update if we made a mistake
            # Get current prediction
            scores = np.dot(self.weights, x)
            pred = np.argmax(scores)
            
            # Calculate current gradient (flattened for L-BFGS)
            current_grad = np.zeros_like(self.weights)
            current_grad[y] = -x      # Increase correct class score
            current_grad[pred] = x    # Decrease predicted class score
            current_grad_flat = current_grad.flatten()
            
            # Store previous state for L-BFGS
            if self.prev_grad is not None and self.prev_weights is not None:
                # Calculate s (step) and y (gradient difference)
                s = self.weights.flatten() - self.prev_weights
                y_diff = current_grad_flat - self.prev_grad
                
                # Calculate rho
                sy = np.dot(s, y_diff)
                if abs(sy) > 1e-10:  # Avoid division by zero
                    rho = 1.0 / sy
                    
                    # Update memory
                    self.s_memory.append(s.copy())
                    self.y_memory.append(y_diff.copy())
                    self.rho_memory.append(rho)
                    
                    # Keep only recent memory
                    if len(self.s_memory) > self.memory_size:
                        self.s_memory.pop(0)
                        self.y_memory.pop(0)
                        self.rho_memory.pop(0)
            
            # Calculate L-BFGS direction
            direction_flat = self._calculate_lbfgs_direction(current_grad_flat)
            direction = direction_flat.reshape(self.weights.shape)
            
            # Store current state
            self.prev_weights = self.weights.flatten().copy()
            self.prev_grad = current_grad_flat.copy()
            
            # Update weights
            self.weights -= self.learning_rate * direction
    
    def _calculate_lbfgs_direction(self, grad: np.ndarray) -> np.ndarray:
        """Calculate L-BFGS search direction."""
        if not self.s_memory:
            return grad  # First iteration, use gradient
        
        # L-BFGS two-loop recursion
        q = grad.copy()
        alphas = []
        
        # First loop (backward)
        for i in range(len(self.s_memory) - 1, -1, -1):
            alpha = self.rho_memory[i] * np.dot(self.s_memory[i], q)
            alphas.append(alpha)
            q -= alpha * self.y_memory[i]
        
        # Initial Hessian approximation (scaled identity)
        if len(self.s_memory) > 0:
            gamma = np.dot(self.s_memory[-1], self.y_memory[-1]) / np.dot(self.y_memory[-1], self.y_memory[-1])
            gamma = max(gamma, 1e-8)  # Ensure positive
            r = gamma * q
        else:
            r = q
        
        # Second loop (forward)
        alphas.reverse()
        for i in range(len(self.s_memory)):
            beta = self.rho_memory[i] * np.dot(self.y_memory[i], r)
            r += (alphas[i] - beta) * self.s_memory[i]
        
        return r


class FogoMemoryPair(OnlineAlgorithm):
    """
    Fogo Memory Pair wrapper that adapts StreamNewtonMemoryPair to the OnlineAlgorithm interface.
    
    This wrapper converts the classification problem to a regression problem by treating
    each class as a separate target (one-hot encoding) and using the Fogo memory pair
    for each class dimension.
    """
    
    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.01, 
                 lam: float = 0.01, eps_total: float = 1.0, delta_total: float = 1e-5,
                 max_deletions: int = 100, seed: int = 42):
        super().__init__(n_features, n_classes, seed)
        self.learning_rate = learning_rate
        self.lam = lam
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # For multi-class classification, we create one StreamNewtonMemoryPair per class
        # This effectively treats it as n_classes separate binary classification problems
        self.memory_pairs = []
        for i in range(n_classes):
            memory_pair = StreamNewtonMemoryPair(
                dim=n_features,
                lam=lam,
                eps_total=eps_total,
                delta_total=delta_total,
                max_deletions=max_deletions
            )
            self.memory_pairs.append(memory_pair)
        
        self.weights = np.zeros((n_classes, n_features))
        self._update_weights()
    
    def _update_weights(self):
        """Update the weights matrix from the memory pairs."""
        for i, memory_pair in enumerate(self.memory_pairs):
            self.weights[i] = memory_pair.theta
    
    def predict(self, x: np.ndarray) -> int:
        """Make a prediction using the current weights."""
        # Update weights from memory pairs
        self._update_weights()
        
        # Compute scores for each class
        scores = np.dot(self.weights, x)
        return np.argmax(scores)
    
    def update(self, x: np.ndarray, y: int, loss: float) -> None:
        """Update the memory pairs with new sample."""
        if loss > 0:  # Only update if we made a mistake
            # One-hot encoding: target is 1 for correct class, 0 for others
            for i, memory_pair in enumerate(self.memory_pairs):
                target = 1.0 if i == y else 0.0
                memory_pair.insert(x, target)
            
            # Update weights
            self._update_weights()


def get_algorithm(algo_name: str, n_features: int, n_classes: int, seed: int = 42) -> OnlineAlgorithm:
    """Get the appropriate algorithm instance."""
    if algo_name == "memorypair":
        return FogoMemoryPair(n_features, n_classes, seed=seed)
    elif algo_name == "sgd":
        return OnlineSGD(n_features, n_classes, seed=seed)
    elif algo_name == "adagrad":
        return AdaGrad(n_features, n_classes, seed=seed)
    elif algo_name == "ons":
        return OnlineNewtonStep(n_features, n_classes, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")