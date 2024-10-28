import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate synthetic data
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# Add bias term
X_b = np.c_[np.ones((m, 1)), X]  # add x0 = 1 to each instance

# SGD parameters
theta_path_sgd = []
cost_values = []
n_epochs = 3
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

# Initialize theta
theta = np.random.randn(2, 1)

# Perform Stochastic Gradient Descent
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta.flatten())
        # Calculate and store the cost after each iteration
        cost = np.mean((X_b @ theta - y) ** 2)
        cost_values.append(cost)

# Convert to numpy arrays for easy indexing
theta_path_sgd = np.array(theta_path_sgd)
cost_values = np.array(cost_values)

# Prepare the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_xlim(0, 2)
ax1.set_ylim(0, 15)
ax1.set_title("Stochastic Gradient Descent Animation")
ax1.set_xlabel("X")
ax1.set_ylabel("y")
ax2.set_title("Cost Over Iterations")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Cost")

# Set y-limits for the cost plot
ax2.set_ylim(0, np.max(cost_values) * 1.1)

# Add grid to cost plot
ax2.grid(True)

# Plot the original data points
ax1.scatter(X, y, color='blue', label='Data points')
line, = ax1.plot([], [], color='red', label='Best fit line')
cost_line, = ax2.plot([], [], color='orange', label='Cost', linewidth=2)
cost_text = ax1.text(0.1, 14, '', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Animation function
def update(frame):
    if frame < len(theta_path_sgd):
        theta_current = theta_path_sgd[frame]
        # Update the line to reflect the new theta
        x_vals = np.array([[0], [2]])  # X values for line
        y_vals = theta_current[0] + theta_current[1] * x_vals  # Linear equation
        line.set_data(x_vals, y_vals)
        
        # Update the cost line
        cost_line.set_data(np.arange(len(cost_values)), cost_values)  # Plot all cost values
        cost_line.set_xdata(np.arange(len(cost_values)))  # Update x data to show all cost values

        cost = cost_values[frame]  # Current cost
        cost_text.set_text(f'Cost: {cost:.2f}')

        # Update the x-limits dynamically to show the most recent iterations
        ax2.set_xlim(0, min(len(cost_values), frame + 10))  # Show a maximum of 10 iterations in view

    return line, cost_line, cost_text

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(theta_path_sgd), blit=True, repeat=False)

# Show the plot
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.show()

# To save the animation as a GIF, uncomment the following line:
ani.save('sgd_animation_cost.gif', writer='imagemagick')
