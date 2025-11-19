import numpy as np
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import zivot_andrews
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

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
    plt.plot(data.values, label='Actual',  marker='o')
    plt.plot(range(start_index, start_index + len(predictions)), predictions, color='red', label='Predicted',  marker='o')
    plt.axvline(x=start_index, color='gray', linestyle='--', label='Forecast start')
    plt.legend()
    plt.show()

    test_series = pd.Series(test.flatten() if hasattr(test, 'flatten') else test)
    pred_series = pd.Series(predictions)

    metrics = evaluate_forecasts(test_series, pred_series)
    print("__________________________________")

    return {"Predictions": predictions,
            "Metrics": metrics}





def ljungbox_test(series, lags=None, **kw):
    """
    Wykonuje test Ljung-Box na resztach szeregu czasowego.

    Parametry:
    - series: szereg czasowy lub reszty modelu
    - lags: liczba opóźnień do sprawdzenia (domyślnie min(10, len(series)//5))
    - **kw: dodatkowe argumenty przekazywane do acorr_ljungbox

    Wyświetla statystyki testu i wnioski o niezależności szumu.
    """
    if lags is None:
        lags = min(10, len(series)//5)

    lb_test = acorr_ljungbox(series, lags=lags, return_df=True, **kw)

    print(f"Ljung-Box Test (lags = {lags}):")
    for lag in lb_test.index:
        stat = lb_test.loc[lag, 'lb_stat']
        p_val = lb_test.loc[lag, 'lb_pvalue']
        print(f"Lag {lag}: LB Statistic = {stat:.4f}, p-value = {p_val:.4f}")
        if p_val < 0.05:
            print(RED + "   Conclusion: Autocorrelation present (Non-random)" + RESET)
        else:
            print(GREEN + "   Conclusion: No significant autocorrelation (Random)" + RESET)
    print("\n")





def durbin_watson_test(series):

    dw_stat = durbin_watson(series)
    print(f"Durbin-Watson Statistic: {dw_stat:.4f}")

    if dw_stat < 1.5:
        print(RED + "Conclusion: Positive autocorrelation present" + RESET)
    elif dw_stat > 2.5:
        print(RED + "Conclusion: Negative autocorrelation present" + RESET)
    else:
        print(GREEN + "Conclusion: No significant autocorrelation" + RESET)
    print("\n")