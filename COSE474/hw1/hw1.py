import numpy as np
from scipy.optimize import minimize

# There are two PARTs in this code that you need to complete

# Mean Square Error function
def MSE(x, A, b):
    # parameter:
    # x: vector with shape (n,)
    # A: matrix with shape (m,n)
    # b: vector with shape (m,)
    ##############################
    # PART 1: completing MSE function
    ##############################
    res = A@x.T - b
    err = (res**2).sum() / len(b)
    # You need to fill in here.
    # Your function must return the MSE between Ax and b
    
    return err

# size of matrix A (m by n) and b (m by 1)
m = 1000
n = 10

# for random generation of data points
mu = 0
sig = 1

mu_noise = 0
sig_noise = 0.1

# true parameter vector
x_true = np.random.normal(mu, sig, n)

# create matrices for least squares function
A = np.c_[np.ones((m, 1)), np.random.normal(mu, sig, (m, n - 1))]
b = A @ x_true + np.random.normal(mu_noise, sig_noise, m)

# initial point for minimize function
x0 = np.random.normal(mu, sig, n)

##############################
# PART 2: Finding estimate using minimize function
##############################
# use minimize function to find the best parameter estimate
# you need to properly use minimize function below
estim = minimize(MSE, x0, args=(A, b))

print('solution from minimize:', estim.x)
print('true x', x_true)

# show error between true x and your estimate
print('error percentage:', np.linalg.norm(x_true - estim.x) / np.linalg.norm(x_true) * 100, '%')

