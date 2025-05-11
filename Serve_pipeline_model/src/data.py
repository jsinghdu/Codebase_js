import numpy as np

# Define data set


def data_sample(n=20) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(0, 10, n).reshape(-1, 1)
    return x, y(x)


def y(x):
    return 2 * x + 0.5 * x**2 - 0.07 * x**3 + 1.5 * np.random.randn(*x.shape)
