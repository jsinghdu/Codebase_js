import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def polynomial(n):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=n)),
        ("linear", linear_model.LinearRegression(fit_intercept=False)),
    ])


def plot_figure(x, y, x_test, y_test, r2=None, val_r2=None, n=None, model=None):
    fig = plt.figure(figsize=(4, 4))
    plt.scatter(x, y)
    plt.scatter(x_test, y_test)
    if model:
        plt.plot(
            np.arange(0, 10, 0.1),
            model.predict(np.arange(0, 10, 0.1).reshape(-1, 1)),
        )
    if r2:
        plt.title(f"Polynomial order {n}: val_R2={val_r2:.4f}, R2={r2:.4f}")
    return fig


def plot_performance_figure(poly_orders, r2, val_r2):
    fig = plt.figure(figsize=(4, 4))
    plt.plot(poly_orders, np.abs(1 - np.array(r2)), label="R2", linewidth=3, alpha=0.5)
    plt.plot(
        poly_orders,
        np.abs(1 - np.array(val_r2)),
        label="val_R2",
        linewidth=3,
        alpha=0.5,
    )
    plt.title("|1-R2| vs. polynomial order")
    plt.legend()
    plt.yscale("log")
    return fig
