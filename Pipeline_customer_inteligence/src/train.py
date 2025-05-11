# This script will do stuff
import os
import secrets
import textwrap

import matplotlib.pyplot as plt
import mlflow
from mlflow.models.signature import infer_signature

from .data import data_sample
from .utils import plot_figure, plot_performance_figure, polynomial


def train(
    poly_order_range: list[int] = [1, 9], artifact_path: str = "artifacts"
) -> tuple[str, str, str]:
    experiment_name = f"{os.environ['MLFLOW_NAME_PREFIX']}.example"
    mlflow.set_experiment(experiment_name=experiment_name)

    exp_description = textwrap.dedent("""
        # Line fitting example
        This is a line regression problem for example puprposes.
        We will be trying to create a model that can predict

        >      2x + 0.5x^2 - 0.07x^3 + 1.5*random()

        This experiment will try fitting polynomials of different order to the data and then
        we will choose the best model based on the validation R2 score.

        ## Data
        `x` is a random uniform distribution between 0 and 10.
    """)
    metric_name = "validation_r2"

    x, y = data_sample(20)
    x_test, y_test = data_sample(5)

    r2 = []
    val_r2 = []
    poly_orders = list(range(poly_order_range[0], poly_order_range[1] + 1))
    with mlflow.start_run(
        run_name=f"poly-run-{secrets.token_urlsafe(3)}",
        tags={"tag_key": "tag_value"},
        description=exp_description,
    ) as run:
        for n in poly_orders:
            with mlflow.start_run(run_name=f"poly-order-{n}", nested=True):
                model = polynomial(n)
                model.fit(x, y)

                r2.append(model.score(x, y))
                val_r2.append(model.score(x_test, y_test))
                fig = plot_figure(x, y, x_test, y_test, r2[-1], val_r2[-1], n, model)
                plt.show()

                mlflow.log_param("polynomial_order", n)
                mlflow.log_metric("r2", r2[-1])
                mlflow.log_metric(metric_name, val_r2[-1])
                mlflow.log_figure(fig, "regression_fit.png")

                sig = infer_signature(x_test, model.predict(x_test))
                mlflow.sklearn.log_model(model, artifact_path, signature=sig)

        fig = plot_performance_figure(poly_orders, r2, val_r2)
        mlflow.log_figure(fig, "performance_vs_order.png")

    return run.info.run_id, experiment_name, metric_name
