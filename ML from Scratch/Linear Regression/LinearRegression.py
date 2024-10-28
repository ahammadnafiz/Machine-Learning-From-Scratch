import numpy as np
from math import ceil
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError
from typing import Optional, Literal, Tuple, Union

class LinearRegressionScratch(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        tolerance: float = 1e-4,
        patience: int = 10,
        batch_size: Optional[int] = None,
        decay: float = 0.99,
        validation_split: float = 0.2,
        regularization: float = 0.0,
        regularization_type: Literal['L1', 'L2'] = 'L2',
        decay_type: Literal['exponential', 'inverse_time'] = 'exponential',
        verbose: int = 0,
        clip_value: Optional[float] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize the Linear Regression model.
        
        Args:
            learning_rate: Initial learning rate for gradient descent
            epochs: Maximum number of training epochs
            tolerance: Minimum change in validation loss to qualify as improvement
            patience: Number of epochs to wait for improvement before early stopping
            batch_size: Size of mini-batches (None for full-batch)
            decay: Learning rate decay factor
            validation_split: Fraction of data to use for validation
            regularization: Regularization strength
            regularization_type: Type of regularization ('L1' or 'L2')
            decay_type: Type of learning rate decay ('exponential' or 'inverse_time')
            verbose: Print training progress every verbose epochs
            clip_value: Maximum absolute value for gradient clipping
            random_state: Random seed for reproducibility
        """
        self._validate_inputs(
            learning_rate, epochs, tolerance, patience, 
            validation_split, regularization, decay
        )
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.patience = patience
        self.batch_size = batch_size
        self.decay = decay
        self.validation_split = validation_split
        self.regularization = regularization
        self.regularization_type = regularization_type
        self.decay_type = decay_type
        self.verbose = verbose
        self.clip_value = clip_value
        self.random_state = random_state
        
        # Initialize model parameters
        self.w = None
        self.b = None
        self.history = []
        self.val_history = []
        self._is_fitted = False

    def _validate_inputs(
        self,
        learning_rate: float,
        epochs: int,
        tolerance: float,
        patience: int,
        validation_split: float,
        regularization: float,
        decay: float
    ) -> None:
        """Validate input parameters."""
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if not isinstance(patience, int) or patience < 0:
            raise ValueError("patience must be a non-negative integer")
        if not isinstance(validation_split, float) or not 0 <= validation_split < 1:
            raise ValueError("validation_split must be a float between 0 and 1")
        if not isinstance(regularization, (int, float)) or regularization < 0:
            raise ValueError("regularization must be non-negative")
        if not isinstance(decay, float) or not 0 < decay < 1:
            raise ValueError("decay must be a float between 0 and 1")

    def _compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        include_regularization: bool = True
    ) -> Tuple[float, np.ndarray, float]:
        """
        Compute the loss, gradients for weights and bias.
        
        Args:
            X: Input features
            y: Target values
            include_regularization: Whether to include regularization term
            
        Returns:
            tuple: (loss, weight gradients, bias gradient)
        """
        y_pred = np.dot(X, self.w) + self.b  # Using direct computation instead of predict()
        m = X.shape[0]
        
        # Compute MSE loss and gradients
        loss = np.mean((y - y_pred) ** 2)
        dw = (-2 / m) * np.dot(X.T, (y - y_pred))
        db = (-2 / m) * np.sum(y - y_pred)
        
        if include_regularization and self.regularization > 0:
            if self.regularization_type == 'L2':
                reg_loss = 0.5 * self.regularization * np.sum(self.w ** 2)
                reg_dw = self.regularization * self.w
            else:  # L1
                reg_loss = self.regularization * np.sum(np.abs(self.w))
                reg_dw = self.regularization * np.sign(self.w)
            
            loss += reg_loss
            dw += reg_dw
            
        return loss, dw, db

    def _learning_schedule(self, t: int) -> float:
        """
        Compute learning rate based on schedule type and current iteration.
        
        Args:
            t: Current iteration number
            
        Returns:
            float: Current learning rate
        """
        if self.decay_type == 'exponential':
            return self.learning_rate * (self.decay ** t)
        else:  # inverse_time
            t0, t1 = 200, 1000  # hyperparameters for inverse time decay
            return self.learning_rate * t0 / (t + t1)

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> 'LinearRegressionScratch':
        """
        Fit the linear regression model using gradient descent.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            self: The fitted model
        """
        # Convert inputs to numpy arrays if they're lists
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # Input validation
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Ensure we have enough samples for validation
        min_train_samples = 2
        if n_samples < min_train_samples / (1 - self.validation_split):
            raise ValueError(
                f"Not enough samples for validation split. Need at least "
                f"{ceil(min_train_samples / (1 - self.validation_split))} samples."
            )
        
        # Split data into training and validation sets
        split_idx = int((1 - self.validation_split) * n_samples)
        indices = np.random.permutation(n_samples)
        X_train, X_val = X[indices[:split_idx]], X[indices[split_idx:]]
        y_train, y_val = y[indices[:split_idx]], y[indices[split_idx:]]
        
        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0
        self.history = []
        self.val_history = []
        
        best_loss = float('inf')
        no_improvement_count = 0
        batch_size = self.batch_size or n_samples
        n_batches = ceil(len(X_train) / batch_size)
        
        for epoch in range(self.epochs):
            # Shuffle training data
            shuffle_idx = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[shuffle_idx]
            y_train_shuffled = y_train[shuffle_idx]
            
            epoch_loss = 0
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Compute gradients
                batch_loss, dw, db = self._compute_loss(X_batch, y_batch)
                epoch_loss += batch_loss
                
                # Apply gradient clipping if specified
                if self.clip_value is not None:
                    dw = np.clip(dw, -self.clip_value, self.clip_value)
                    db = np.clip(db, -self.clip_value, self.clip_value)
                
                # Update parameters
                lr = self._learning_schedule(epoch * n_batches + i)
                self.w -= lr * dw
                self.b -= lr * db
            
            # Compute average epoch loss and validation loss
            avg_epoch_loss = epoch_loss / n_batches
            val_loss, _, _ = self._compute_loss(X_val, y_val, include_regularization=False)
            
            self.history.append(avg_epoch_loss)
            self.val_history.append(val_loss)
            
            # Early stopping check
            if val_loss < best_loss:
                relative_improvement = (best_loss - val_loss) / best_loss
                if relative_improvement < self.tolerance:
                    no_improvement_count += 1
                else:
                    best_loss = val_loss
                    no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= self.patience:
                if self.verbose > 0:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
                
            if self.verbose > 0 and epoch % self.verbose == 0:
                print(f"Epoch {epoch + 1}, Training Loss: {avg_epoch_loss:.4f}, "
                      f"Validation Loss: {val_loss:.4f}")
        
        self._is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Make predictions for the input features.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        if not self._is_fitted:
            raise NotFittedError(
                "This LinearRegressionScratch instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        
        X = np.array(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if X.shape[1] != len(self.w):
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with {len(self.w)} features"
            )
            
        return np.dot(X, self.w) + self.b

    def score(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:
        """
        Calculate R-squared score for the model.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            float: R-squared score
        """
        if not self._is_fitted:
            raise NotFittedError(
                "This LinearRegressionScratch instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
            
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
            
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - (u / v) if v != 0 else 0.0

    def learning_curve(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot the learning curves showing training and validation loss over epochs.
        
        Args:
            figsize: Figure size as (width, height)
        """
        if not self._is_fitted:
            raise NotFittedError(
                "This LinearRegressionScratch instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
            
        plt.figure(figsize=figsize)
        plt.plot(self.history, label="Training Loss")
        plt.plot(self.val_history, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Learning Curves")
        plt.legend()
        plt.grid(True)
        plt.show()