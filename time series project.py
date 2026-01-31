# ==========================================================
# Advanced Time Series Forecasting with Prophet
# Includes Calibration & Rolling Cross Validation
# ==========================================================

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ==========================================================
# 1. Generate Complex Synthetic Time Series
# ==========================================================

np.random.seed(42)

dates = pd.date_range("2018-01-01", "2023-12-31", freq="D")
n = len(dates)

trend = 0.02 * np.arange(n)

weekly = 5 * np.sin(2 * np.pi * dates.dayofweek / 7)
yearly = 10 * np.sin(2 * np.pi * dates.dayofyear / 365)
daily = 2 * np.sin(2 * np.pi * np.arange(n))

noise = np.random.normal(0, 3, n)
y = trend + weekly + yearly + daily + noise

df = pd.DataFrame({"ds": dates, "y": y})

# ==========================================================
# 2. Add Holiday Effects (>= 5 events)
# ==========================================================

holidays = pd.DataFrame({
    "holiday": "special_event",
    "ds": pd.to_datetime([
        "2019-01-01","2019-12-25",
        "2020-01-01","2020-12-25",
        "2021-01-01","2021-12-25",
        "2022-01-01","2022-12-25",
        "2023-01-01"
    ]),
    "lower_window": 0,
    "upper_window": 1
})
# ==========================================================
# 3. Rolling-Origin Cross Validation Function
# ==========================================================

def rolling_cv(data, holidays, interval_width=0.8,
               cps=0.05, horizon=90, initial=730, step=180):

    metrics = []

    for start in range(initial, len(data) - horizon, step):

        train = data.iloc[:start]
        test = data.iloc[start:start+horizon]

        model = Prophet(
            holidays=holidays,
            interval_width=interval_width,
            changepoint_prior_scale=cps,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )

        model.fit(train)
future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future).tail(horizon)

        # Accuracy
        rmse = np.sqrt(mean_squared_error(test["y"], forecast["yhat"]))

        # Coverage Probability
        coverage = np.mean(
            (test["y"] >= forecast["yhat_lower"]) &
            (test["y"] <= forecast["yhat_upper"])
        )

        # Mean Interval Width
        interval_width_mean = np.mean(
            forecast["yhat_upper"] - forecast["yhat_lower"]
        )

        metrics.append({
            "rmse": rmse,
            "coverage": coverage,
            "interval_width": interval_width_mean
        })

    return pd.DataFrame(metrics)
# ==========================================================
# 4. Initial Model Evaluation
# ==========================================================

results_80 = rolling_cv(df, holidays, interval_width=0.80, cps=0.05)
results_95 = rolling_cv(df, holidays, interval_width=0.95, cps=0.05)

initial_summary = pd.DataFrame({
    "PI Level": ["80%", "95%"],
    "Avg RMSE": [results_80.rmse.mean(), results_95.rmse.mean()],
    "Coverage": [results_80.coverage.mean(), results_95.coverage.mean()],
    "Mean Interval Width": [
        results_80.interval_width.mean(),
        results_95.interval_width.mean()
    ]
})

print("\n===== INITIAL MODEL PERFORMANCE =====")
print(initial_summary)
# ==========================================================
# 5. Calibrated Model (Increase CPS)
# ==========================================================

results_80_cal = rolling_cv(df, holidays, interval_width=0.80, cps=0.15)
results_95_cal = rolling_cv(df, holidays, interval_width=0.95, cps=0.15)

calibrated_summary = pd.DataFrame({
    "PI Level": ["80%", "95%"],
    "Avg RMSE": [results_80_cal.rmse.mean(), results_95_cal.rmse.mean()],
    "Coverage": [results_80_cal.coverage.mean(), results_95_cal.coverage.mean()],
    "Mean Interval Width": [
        results_80_cal.interval_width.mean(),
        results_95_cal.interval_width.mean()
    ]
})

print("\n===== CALIBRATED MODEL PERFORMANCE =====")
print(calibrated_summary)

# ==========================================================
# 6. Final Model Fit (Recommended Configuration)
# ==========================================================

final_model = Prophet(
    holidays=holidays,
    interval_width=0.95,
    changepoint_prior_scale=0.15,
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)

final_model.fit(df)

future = final_model.make_future_dataframe(periods=180)
forecast = final_model.predict(future)

# ==========================================================
# 7. Plot Forecast
# ==========================================================

final_model.plot(forecast)
plt.title("Final Forecast with 95% Prediction Interval")
plt.show()

# ==========================================================
# 8. Plot Model Components (Interpretability)
# ==========================================================

final_model.plot_components(forecast)
plt.show()

