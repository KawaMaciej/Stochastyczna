import numpy as np
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import zivot_andrews
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error

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

    if result[1] < 0.05:
        print(GREEN + "Conclusion: Stationary" + RESET)
        print("\n")
    else:
        print(RED + "Conclusion: Non-stationary" + RESET)
    print("\n")


def diagnose_stationarity(y):
    kpss_test(y, regression='c')
    adf_test(y)
    zivot_andrews_test(y)


def rolling_forecast_arima(series,
                           order=(1,0,0),
                           start_frac=0.6,
                           method='none',
                           enforce_invertibility=False,
                           enforce_stationarity=False,
                           verbose=False):

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    n = len(series)
    if n < 6:
        raise ValueError("Too few observations for rolling ARIMA.")
    if not (0 < start_frac < 1):
        raise ValueError("start_frac must be between 0 and 1.")

    split = int(n * start_frac)
    train = series.iloc[:split].copy()
    test = series.iloc[split:].copy()
    test_index = test.index

    transformer = None
    trend_full = None
    method_lower = method.lower()

    scaler_map = {
        "scaling": PowerTransformer(method='yeo-johnson', standardize=True),
        "standard scaling": StandardScaler(),
        "quantile scaling": QuantileTransformer(output_distribution='normal')
    }


    if "scaling" in method_lower:
        transformer = scaler_map.get(method_lower)
        if transformer is None:
            raise ValueError(f"Unknown scaling method: {method}")
        train_trans = transformer.fit_transform(train.values.reshape(-1, 1)).flatten()

    elif method_lower == "detrending":
        ets = ExponentialSmoothing(
            train.values, trend='add', seasonal=None,
            initialization_method="estimated"
        )
        ets_fit = ets.fit(optimized=True)

        full_idx = series.index

        try:
            trend_pred = ets_fit.predict(start=0, end=n-1)
        except Exception:
            trend_pred = np.concatenate([
                ets_fit.fittedvalues,
                np.repeat(ets_fit.fittedvalues[-1], n - len(ets_fit.fittedvalues))
            ])

        trend_full = pd.Series(trend_pred, index=full_idx)
        transformed = (series - trend_full).values
        train_trans = transformed[:split]

    else:
        train_trans = train.values.copy()


    def transform_value(x, pos):
        if "scaling" in method_lower:
            return float(transformer.transform(np.array(x).reshape(1, -1)).flatten()[0])
        if method_lower == 'detrending':
            return float(x - trend_full.iloc[pos])
        return float(x)

    def inverse_transform_value(y_trans, pos):
        if "scaling" in method_lower:
            return float(transformer.inverse_transform(np.array(y_trans).reshape(1, -1)).flatten()[0])
        if method_lower == 'detrending':
            return float(y_trans + trend_full.iloc[pos])
        return float(y_trans)

    history = list(train_trans)
    predictions = []
    actuals = []

    for t in range(len(test)):
        global_pos = split + t

        try:
            model = ARIMA(history, order=order,
                          enforce_invertibility=enforce_invertibility,
                          enforce_stationarity=enforce_stationarity)
            model_fit = model.fit()
            if t == 0:
                fitted_values = model_fit.fittedvalues
            yhat_trans = model_fit.forecast(steps=1)[0]
            yhat_orig = inverse_transform_value(yhat_trans, global_pos)
        except Exception as e:
            if verbose:
                print(f"ARIMA fit/forecast failed at step {t} (global_pos {global_pos}): {e}")
            yhat_orig = np.nan
            yhat_trans = np.nan

        actual_val = float(test.iloc[t])
        predictions.append(yhat_orig)
        actuals.append(actual_val)

        try:
            actual_trans = transform_value(actual_val, global_pos)
            if np.isfinite(actual_trans):
                history.append(actual_trans)
        except Exception as e:
            if verbose:
                print(f"Transforming actual failed at step {t}: {e}")


    results_df = pd.DataFrame({
        'actual': actuals,
        'forecast': predictions
    }, index=test_index)

    mask = results_df['forecast'].notna()
    if mask.sum() == 0:
        nan_metrics = {'MAE': np.nan, 'RMSE': np.nan, 'SMAPE': np.nan, 'MASE': np.nan, 'R^2': np.nan}
        return results_df, nan_metrics

    metrics = evaluate_forecasts(
        results_df.loc[mask, 'actual'],
        results_df.loc[mask, 'forecast'],
        verbose=verbose
    )
    

    return results_df, metrics, fitted_values





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





def seasonal_decompose_plot(y, figsize=(12,8), plots=False):

    decomposition = seasonal_decompose(y, model='additive', period=4)  

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    if plots:
        plt.figure(figsize=figsize)

        plt.subplot(411)
        plt.plot(y, label='Original')
        plt.legend(loc='upper left')

        plt.subplot(412)
        plt.plot(trend, label='Trend', color='orange')
        plt.legend(loc='upper left')

        plt.subplot(413)
        plt.plot(seasonal, label='Seasonality', color='green')
        plt.legend(loc='upper left')

        plt.subplot(414)
        plt.plot(residual, label='Residuals', color='red')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()
    
    return trend, seasonal, residual




def detrending(y_values, y_index, figsize=(12,8), plots=False):

    ets_model = ExponentialSmoothing(
        y_values,
        trend="add",
        seasonal=None,
        initialization_method="estimated"
    )

    try:
        ets_fit = ets_model.fit(optimized=True)
    except Exception as e:
        raise ValueError(f"ETS failed to fit: {e}")

    trend = ets_fit.fittedvalues

    trend_series = pd.Series(trend, index=y_index)

    y_detrended_vals = y_values - trend_series.values
    y_detrended = pd.Series(y_detrended_vals, index=y_index)

    if plots:
        plt.figure(figsize=figsize)
        plt.plot(y_index, y_values, label='Original', linewidth=2, color='brown')
        plt.plot(trend_series, label='ETS Trend', linestyle='--', linewidth=2, color='orange')
        plt.plot(y_detrended, label='Detrended (Original - Trend)', linewidth=2, color='teal')
        plt.title('ETS Detrending: Removing Additive Trend')
        plt.legend()
        plt.show()

        print(f"Original std:   {np.std(y_values):.4f}")
        print(f"Detrended std:  {np.std(y_detrended_vals):.4f}")

    return y_detrended


def change_data(data, method):
    if method == "detranding":
        data_transformed = detrending(data.values, data.index)

    elif method == "scaling":
        pt = PowerTransformer(method='yeo-johnson')
        data_transformed = pt.fit_transform(data.values.reshape(-1,1)).flatten()

    else:
        data_transformed = data.values


    return data_transformed