import numpy as np
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import zivot_andrews
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def evaluate_forecasts(test, pred, verbose=True):
    mae = np.abs((pred - test)).mean()
    mse = ((pred - test)**2).mean()
    rmse = np.sqrt(mse)

    smape = 100 * np.mean(
        np.abs(pred - test) /
        ((np.abs(test) + np.abs(pred)) / 2)
    )

    naive = test.shift(1).dropna()
    mase = (pred[1:] - test[1:]).abs().mean() / \
        (test[1:] - naive).abs().mean()

    r2 = r2_score(test, pred)
    if verbose:
        print("MAE:", mae)
        print("RMSE:", rmse)
        print("sMAPE (%):", smape)
        print("MASE:", mase)
        print("R² Score:", r2)

    return {"MAE": mae, 
            "RMSE": rmse, 
            "SMAPE": smape, 
            "MASE": mase, 
            "R^2": r2 
            }


def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    if p_value < 0.05:
        print(RED + "Conclusion: Non-stationary" + RESET)
        print("\n")
    else:
        print(GREEN + "Conclusion: Stationary" + RESET)
        print("\n")

def adf_test(series):
    result = adfuller(series)
    print("==== Augmented Dickey-Fuller Test ====")
    print(f"ADF Statistic : {result[0]}")
    print(f"p-value       : {result[1]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key} : {value}")

    if result[1] < 0.05:
        print(GREEN + "Conclusion: Stationary" + RESET)
    else:
        print(RED + "Conclusion: Non-stationary" + RESET)

    print("\n")

def zivot_andrews_test(series):
    result = zivot_andrews(series, maxlag=12)

    print("==== Zivot–Andrews Test (structural break) ====")
    print(f"Test Statistic : {result[0]}")
    print(f"p-value        : {result[1]}")
    print(f"Break at index : {result[2]}")

    crit_vals = result[3]

    print("Critical Values:")

    if isinstance(crit_vals, dict):
        for key, value in crit_vals.items():
            print(f"   {key} : {value}")
    else:
        print("   Critical values not provided in dictionary form.")

    if result[1] < 0.05:
        print(GREEN + "Conclusion: Stationary" + RESET)
        print("\n")
    else:
        print(RED + "Conclusion: Non-stationary" + RESET)
    print("\n")


def rolling_forecast_arima(train, test, order, start_index, dfs, scaler=None):
    data = dfs.copy()
    history = train.tolist()
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=order, enforce_invertibility=False, enforce_stationarity=False)  
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t]) 
    
    if scaler != None:
        data_values = scaler.inverse_transform(data.values.reshape(-1, 1)).flatten()
        data = pd.Series(data_values, index=data.index)
        
        history = scaler.inverse_transform(np.array(history).reshape(-1, 1)).flatten()
        test = scaler.inverse_transform(np.array(test).reshape(-1, 1)).flatten()
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(data.values, label='Actual')
    plt.plot(range(start_index, start_index + len(predictions)), predictions, color='red', label='Predicted')
    plt.axvline(x=start_index, color='gray', linestyle='--', label='Forecast start')
    plt.legend()
    plt.show()

    test_series = pd.Series(test.flatten() if hasattr(test, 'flatten') else test)
    pred_series = pd.Series(predictions)

    metrics = evaluate_forecasts(test_series, pred_series)
    print("__________________________________")

    return {"Predictions": predictions,
            "Metrics": metrics}