from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load the diabetes dataset
diabetes = load_diabetes()
A = diabetes.data
b = diabetes.target

# Gradient Descent function for training only
def gd_train_only(A_train, b_train, learning_rate, xinit, tolerance):
    x = xinit 
    train_errors = []
    while True:
        gradiente = A_train.T @ (A_train @ x - b_train)
        train_error = np.linalg.norm(A_train @ x - b_train)**2 / A_train.shape[0]
        train_errors.append(train_error)
        gradiente_norm = np.linalg.norm(gradiente)
        gdnorm = 2 * (gradiente_norm ** 2)
        if gdnorm <= tolerance:
            break
        x = x - learning_rate * 2 * gradiente
    return x, train_errors

# Gradient Descent function for training and testing
def gd_train_test(A_train, b_train, A_test, b_test, learning_rate, xinit, tolerance):
    x = xinit 
    train_errors = []
    test_errors = []
    while True:
        gradiente = A_train.T @ (A_train @ x - b_train)
        train_error = np.linalg.norm(A_train @ x - b_train)**2 / A_train.shape[0]
        test_error = np.linalg.norm(A_test @ x - b_test)**2 / A_test.shape[0]
        train_errors.append(train_error)
        test_errors.append(test_error)
        gradiente_norm = np.linalg.norm(gradiente)
        gdnorm = 2 * (gradiente_norm ** 2)
        if gdnorm <= tolerance:
            break
        x = x - learning_rate * 2 * gradiente
    return x, train_errors, test_errors

# Part A
xinit = np.zeros(A.shape[1])
learning_rate = 1e-2
tolerance = 1e-6
x_gd, errors_gd = gd_train_only(A, b, learning_rate=learning_rate, xinit=xinit, tolerance=tolerance)

# Plot the error over iterations
plt.figure(1)
plt.plot(errors_gd)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('Error over Iterations')
plt.grid(True)

# Part B
# Split the data into training and test sets
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=1)

# Initialize xinit
xinit = np.zeros(A_train.shape[1])

# Call the gd function with adjusted learning rate and tolerance
x_gd, train_errors_gd, test_errors_gd = gd_train_test(A_train, b_train, A_test, b_test, learning_rate, xinit, tolerance)

# Plot the train and test errors over iterations
plt.figure(2)
plt.plot(train_errors_gd, label='Train Error')
plt.plot(test_errors_gd, label='Test Error')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('Train and Test Error over Iterations')
plt.legend()
plt.grid(True)

# Part C
# Initialize lists to store errors from each run
all_train_errors = []
all_test_errors = []
min_train_errors = []
min_test_errors = []
minrun = float('inf')

# Repeat the process for num_runs times
for run in range(10):
    # Split the data into training and test sets
    A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=run)
    # Initialize xinit
    xinit = np.zeros(A_train.shape[1])
    
    # Call the gd function
    x_gd, train_errors_gd, test_errors_gd = gd_train_test(A_train, b_train, A_test, b_test, learning_rate, xinit, tolerance)
    
    if train_errors_gd[-1] < minrun:
        minrun = train_errors_gd[-1]
        min_train_errors = train_errors_gd
        min_test_errors = test_errors_gd

    # Store the errors
    all_train_errors.append(train_errors_gd)
    all_test_errors.append(test_errors_gd)

# Pad the error lists to make them of equal length
max_length = max(len(errors) for errors in all_train_errors)
padded_train_errors = [errors + [errors[-1]] * (max_length - len(errors)) for errors in all_train_errors]
padded_test_errors = [errors + [errors[-1]] * (max_length - len(errors)) for errors in all_test_errors]

# Compute the average errors
average_train_errors = np.mean(padded_train_errors, axis=0)
average_test_errors = np.mean(padded_test_errors, axis=0)

# Plot the average train and test errors over iterations
plt.figure(3)
plt.plot(average_train_errors, label='Avg Train Error')
plt.plot(average_test_errors, label='Avg Test Error')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('Avg Train and Test Error over Iterations')
plt.legend()
plt.grid(True)

# Plot the minimum train and corresponding test errors over iterations
plt.figure(4)
plt.plot(min_train_errors, label='Min Train Error')
plt.plot(min_test_errors, label='Test Error')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('Min Train and Test Error over Iterations')
plt.legend()
plt.grid(True)
plt.show()
