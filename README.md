# Linear Regression Implementation

## Goal
Implement linear regression to predict house prices.

## Dataset
Use the California Housing dataset from sklearn.datasets.

## Tools
- NumPy
- Matplotlib
- Standard Python

## Deliverables
- Non-vectorized implementation
- Vectorized implementation
- Plots of predictions and training loss

## Data Preparation
- Load the dataset (e.g., `load_boston()` or `fetch_california_housing()`)
- Normalize features to zero mean and unit variance
- Split data: 70% training, 15% validation, 15% testing
- Shuffle data using `np.random.permutation()` before splitting

## Implement Gradient Descent (Non-Vectorized)
- Loop through each training example
- Compute prediction hθ(x)
- Accumulate gradients manually
- Update parameters using the rule: θj := θj - α · ∂J(θ)/∂θj
- Track loss at each epoch and print sample predictions

## Implement Gradient Descent (Vectorized)
- Replace all loops with NumPy matrix operations
- Use formulas:
  - predictions = Xθ
  - ∇J(θ) = (1/m) X^T(Xθ - y)
- Validate that vectorized version gives the same result as the loop-based one
- Compare efficiency

## Plotting and Evaluation
- Plot training loss vs. epoch
- Plot predictions vs. actual values (on validation and test sets)
- Evaluate performance using Mean Squared Error (MSE) on validation and test sets
- Compare results of both implementations
- Save all plots as PNGs