import numpy as np

# Function to numerically compute the Jacobian of any function f
def compute_jacobian(f, x, epsilon=1e-6):
    """
    Compute the Jacobian matrix of the function f at a point x numerically.

    Parameters:
    f: Function that takes in the parameter vector and returns the model output.
    x: Parameter vector at which to compute the Jacobian.
    epsilon: Small perturbation used to calculate finite differences.

    Returns:
    J: Jacobian matrix.
    """
    n = len(x)  # Number of parameters
    m = len(f(x))  # Number of outputs (depends on the shape of f(x))
    
    J = np.zeros((m, n))  # Initialize the Jacobian matrix
    
    for i in range(n):  # For each parameter
        x_perturbed = np.copy(x)
        x_perturbed[i] += epsilon  # Perturb the i-th parameter
        
        # Compute the finite difference (f(x + epsilon) - f(x)) / epsilon
        J[:, i] = (f(x_perturbed) - f(x)) / epsilon
    
    return J

# Levenberg-Marquardt optimization function using numerical Jacobian
def levenberg_marquardt(f, x0, y_data, max_iter=100, tol=1e-6, lam=1e-3):
    """
    Levenberg-Marquardt optimization algorithm using a numerically computed Jacobian.
    
    Parameters:
    f: The model function to be optimized.
    x0: Initial guess for the parameters.
    y_data: Actual data to fit the model to.
    max_iter: Maximum number of iterations.
    tol: Convergence tolerance (when the update is smaller than this, we stop).
    lam: Initial damping factor (lambda).

    Returns:
    x: Optimized parameters.
    """
    x = x0  # Start with initial guess
    n = len(x0)  # Number of parameters

    for i in range(max_iter):
        # Compute the residuals (difference between model output and actual data)
        residuals = f(x) - y_data
        
        # Numerically compute the Jacobian matrix
        J = compute_jacobian(f, x)
        
        # Compute the gradient (J^T * residuals)
        grad = J.T @ residuals
        
        # Compute the Hessian approximation (J^T * J)
        H = J.T @ J
        
        # Levenberg-Marquardt update rule
        I = np.eye(n)  # Identity matrix
        A = H + lam * I  # Add regularization term (damping factor)
        
        # Compute the parameter update step
        delta = np.linalg.inv(A) @ grad
        
        # Update the parameters
        x_new = x - delta
        
        # Check for convergence (if the update is small enough)
        if np.linalg.norm(delta) < tol:
            print(f"Converged at iteration {i+1}")
            break
        
        # Check if the new parameters reduce the error (using residuals)
        new_residuals = f(x_new) - y_data
        if np.linalg.norm(new_residuals) < np.linalg.norm(residuals):
            # Accept the new parameters and decrease lambda (for faster convergence)
            x = x_new
            lam *= 0.9
        else:
            # Reject the new parameters and increase lambda (for more exploration)
            lam *= 10
        
    return x

# Example usage:
if __name__ == "__main__":
    # Example model: A simple exponential function to fit (f(x) = A * exp(B * x))
    def model(params):
        A, B = params
        x_data = np.linspace(0, 4, 10)  # Example input data
        return A * np.exp(B * x_data)  # Model output based on parameters A and B

    # Generate example data with noise (to simulate a noisy dataset)
    true_params = [2.0, 0.5]  # True parameters A=2.0, B=0.5
    x_data = np.linspace(0, 4, 10)
    y_data = model(true_params) + np.random.normal(0, 0.2, len(x_data))  # Add noise

    # Initial guess for the parameters
    initial_guess = [1.0, 0.1]

    # Perform Levenberg-Marquardt optimization
    optimized_params = levenberg_marquardt(model, initial_guess, y_data)
    print("Optimized Parameters:", optimized_params)
