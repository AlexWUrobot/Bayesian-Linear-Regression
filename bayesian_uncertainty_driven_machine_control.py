import numpy as np
import matplotlib.pyplot as plt

# === Ground truth: a perfectly linear motion (e.g., CNC feed axis) ===
true_a, true_b = 1.0, 0.2
noise_std = 0.05

# === Prior belief: unknown slope and offset ===
mu_prior = np.array([0.0, 0.0])     # [b, a]
Sigma_prior = np.eye(2) * 1.0       # large uncertainty
sigma_n = noise_std

# === Simulation data (commanded vs measured positions) ===
X_data = np.linspace(0, 1, 20)
Y_data = true_a * X_data + true_b + np.random.randn(len(X_data)) * noise_std

# === History trackers ===
mu_history = [mu_prior]
sigma_history = []
speed_history = []

# === Control thresholds ===
threshold = 0.15  # uncertainty limit for safe operation

for i in range(len(X_data)):
    x_i = np.array([1.0, X_data[i]])
    y_i = Y_data[i]

    # Step 1: Bayesian update (posterior mean & covariance)
    Sigma_post = np.linalg.inv(
        np.linalg.inv(Sigma_prior) + (1 / sigma_n**2) * np.outer(x_i, x_i)
    )
    mu_post = Sigma_post @ (
        np.linalg.inv(Sigma_prior) @ mu_prior + (1 / sigma_n**2) * x_i * y_i
    )

    # Step 2: Compute posterior uncertainty (σ)
    sigma_post = np.sqrt(np.diag(Sigma_post)).mean()
    sigma_history.append(sigma_post)

    # Step 3: Uncertainty-driven control
    if sigma_post < threshold:
        control_mode = "Normal ✅"
        speed = 1.0
    elif sigma_post < 2 * threshold:
        control_mode = "Caution ⚠️"
        speed = 0.5
    else:
        control_mode = "Stop 🛑"
        speed = 0.0

    speed_history.append(speed)
    print(f"Step {i+1:2d}: σ={sigma_post:.3f}, Mode={control_mode}, Speed={speed}")

    # Step 4: Update prior for next iteration
    mu_prior, Sigma_prior = mu_post, Sigma_post
    mu_history.append(mu_post)

# === Visualization ===
steps = np.arange(1, len(X_data)+1)

plt.figure(figsize=(10,5))
plt.plot(steps, sigma_history, label="Posterior σ (uncertainty)", marker='o')
plt.axhline(threshold, color='g', linestyle='--', label='Safe threshold')
plt.axhline(2*threshold, color='r', linestyle='--', label='Critical threshold')
plt.ylabel("Uncertainty σ")
plt.xlabel("Observation step")
plt.legend()
plt.title("Bayesian Uncertainty Shrinks Over Time")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.step(steps, speed_history, where='post', label="Feed rate (control output)")
plt.ylabel("Normalized Feed Rate (0–1)")
plt.xlabel("Observation step")
plt.title("Uncertainty-Driven Feed Rate Control")
plt.grid(True)
plt.legend()
plt.show()
