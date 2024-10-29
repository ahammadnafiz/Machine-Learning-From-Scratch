import numpy as np
from sklearn.linear_model import LinearRegression
from LinearRegression import LinearRegressionScratch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Test the custom implementation
# ==============================

#Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset with higher noise
def generate_dataset(n_samples=500, n_features=3, noise=0.132):
    """
    Generate a synthetic dataset with linear relationships and higher noise.
    """
    true_coefficients = np.random.randn(n_features)
    true_intercept = 1.1 * np.random.randn()
    
    # Generate feature matrix with some non-linear relationships
    X = 1.1 * np.random.randn(n_samples, n_features)

    # Add polynomial features (squared terms) for non-linearity
    X_poly = np.column_stack((X** 2, 0.5 * X ** 2 + X + 0.3))
    
    # Generate target values with higher noise
    y = np.dot(X_poly, np.concatenate((true_coefficients, np.random.randn(n_features)))) + true_intercept
    
    # Add different noise distributions (e.g., uniform noise)
    y += np.random.normal(0, noise, size=n_samples) + np.random.uniform(-0.1, 0.1, size=n_samples)
    
    return X_poly, y, true_coefficients, true_intercept

# Generate data
X, y, true_coef, true_intercept = generate_dataset(noise=0.5)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train custom implementation with improved hyperparameters
custom_model = LinearRegressionScratch(
    learning_rate=0.1,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    tolerance=1e-5,
    patience=80,
    decay=0.995,
    regularization=1e-4,
    regularization_type='L2',
    decay_type='exponential',
    verbose=100,
    clip_value=1.1,
    random_state=42
)
custom_model.fit(X_train_scaled, y_train)

# Train scikit-learn implementation
sklearn_model = LinearRegression()
sklearn_model.fit(X_train_scaled, y_train)

# Make predictions
custom_pred = custom_model.predict(X_test_scaled)
sklearn_pred = sklearn_model.predict(X_test_scaled)

# Calculate metrics
custom_r2 = r2_score(y_test, custom_pred)
sklearn_r2 = r2_score(y_test, sklearn_pred)
custom_mse = mean_squared_error(y_test, custom_pred)
sklearn_mse = mean_squared_error(y_test, sklearn_pred)

# Print results
print("\nResults Comparison:")
print("-" * 50)
print(f"Custom Implementation:")
print(f"R² Score: {custom_r2:.4f}")
print(f"MSE: {custom_mse:.4f}")
print(f"\nScikit-learn Implementation:")
print(f"R² Score: {sklearn_r2:.4f}")
print(f"MSE: {sklearn_mse:.4f}")

# Plot predictions vs actual values
plt.figure(figsize=(12, 5))

# Custom model predictions
plt.subplot(1, 2, 1)
plt.scatter(y_test, custom_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Custom Implementation\nPredictions vs Actual')

# Scikit-learn model predictions
plt.subplot(1, 2, 2)
plt.scatter(y_test, sklearn_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scikit-learn Implementation\nPredictions vs Actual')

plt.tight_layout()
plt.show()

# Plot learning curves for custom implementation
custom_model.learning_curve()

# Compare coefficients
print("\nCoefficient Comparison:")
print("-" * 50)
print("True coefficients:", true_coef)
print("Custom model coefficients (scaled):", custom_model.w)
print("Scikit-learn coefficients (scaled):", sklearn_model.coef_)
print("\nIntercept Comparison:")
print("-" * 50)
print("True intercept:", true_intercept)
print("Custom model intercept:", custom_model.b)
print("Scikit-learn intercept:", sklearn_model.intercept_)