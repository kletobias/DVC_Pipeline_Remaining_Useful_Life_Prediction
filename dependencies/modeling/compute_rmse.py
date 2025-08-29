import numpy as np
from sklearn.metrics import mean_squared_error


def compute_rmse(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

    Returns:
    Tuple[float,Optional[list]]: The RMSE value.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if rmse.ndim > 0:
        rmse_list = rmse.tolist()
        rmse = np.mean(rmse)
    else:
        rmse_list = None

    rmse = np.round(rmse, 2)

    return rmse, rmse_list
