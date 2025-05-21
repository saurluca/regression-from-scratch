# Linear Regression Implementation

## Goal
Implement linear regression to predict house prices.

## Dataset
Use the California Housing dataset from sklearn.datasets.

## Installation

1. Install uv (if not already installed), [uv docs](https://docs.astral.sh/uv/getting-started/installation/)
:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone this repository and navigate to it:
```bash
git clone <repository-url>
cd regression-from-scratch
```

3. Install dependencies and create virtual environment:
```bash
uv sync
```

## Running the Code

To run any Python script in the project:
```bash
uv run <script-name>.py
```

For example, to run the main implementation:
```bash
uv run main.py
```

## Tools
- NumPy
- Matplotlib
- Standard Python

## Deliverables
- Non-vectorized implementation
- Vectorized implementation
- Plots of predictions and training loss

## Data Preparation
- Load the dataset (fetch_california_housing())
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