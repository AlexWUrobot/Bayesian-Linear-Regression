import numpy as np
import matplotlib.pyplot as plt

# Ground truth (actual machine linear motion)
true_a, true_b = 1.0, 0.2
noise_std = 0.05

# Prior belief about parameters θ = [b, a]
mu_prior = np.array([0.0, 0.0])
Sigma_prior = np.eye(2) * 1.0   # large uncertainty

# Observation noise
sigma_n = noise_std

# Record evolution
mu_history = [mu_prior]
sigma_history = []

# Simulated trajectory inputs (e.g., commanded x positions)
X_data = np.linspace(0, 1, 20)
Y_data = true_a * X_data + true_b + np.random.randn(len(X_data)) * noise_std

for i in range(len(X_data)):
    x_i = np.array([1.0, X_data[i]])   # feature vector [bias, x]
    y_i = Y_data[i]

    # Posterior covariance update
    Sigma_post = np.linalg.inv(
        np.linalg.inv(Sigma_prior) + (1 / sigma_n**2) * np.outer(x_i, x_i)
    )

    # Posterior mean update
    mu_post = Sigma_post @ (
        np.linalg.inv(Sigma_prior) @ mu_prior + (1 / sigma_n**2) * x_i * y_i
    )

    # Store and roll forward
    mu_history.append(mu_post)
    Sigma_prior = Sigma_post
    mu_prior = mu_post
    sigma_history.append(np.sqrt(np.diag(Sigma_post)).mean())

# Plot parameter uncertainty evolution
plt.figure(figsize=(7, 4))
plt.plot(sigma_history, label="Posterior σ (avg over parameters)")
plt.xlabel("Observation step")
plt.ylabel("Uncertainty σ")
plt.title("Posterior Uncertainty Decreases as More Data Arrives")
plt.legend()
plt.grid(True)
plt.show()

# Plot final line vs. noisy observations
plt.figure(figsize=(7, 4))
plt.scatter(X_data, Y_data, label="Sensor data", color="orange")
plt.plot(X_data, true_a*X_data + true_b, 'k--', label="True line")
mu_final = mu_history[-1]
plt.plot(X_data, mu_final[1]*X_data + mu_final[0], 'b', label="Bayesian fit")
plt.xlabel("Commanded position (X)")
plt.ylabel("Measured position (Y)")
plt.title("Bayesian Linear Regression of Tool Trajectory")
plt.legend()
plt.grid(True)
plt.show()

