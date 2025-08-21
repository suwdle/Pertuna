import numpy as np
def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float); y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=1e-9)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))
