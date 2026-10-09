# Daily Temperature Forecasting in Japan with Stacked LSTM

A time series model that forecasts daily average temperature in Japan from the previous 100 days, using a five-layer stacked LSTM in TensorFlow/Keras.

![Average temperature in Japan](https://github.com/harrymardika/DailyTemperature-LSTM/assets/130530985/30623ef6-b414-4c9f-951c-c180de12f4f1)

## Results

| Metric (min-max scaled) | Value |
|---|---|
| Training MAE | 0.0497 |
| Validation MAE | 0.0440 |
| Target | MAE below 0.05 on both sets |
| Epochs | 34 of 100 (stopped by the MAE threshold callback) |

![Actual vs. predicted temperature on the test set](https://github.com/harrymardika/DailyTemperature-LSTM/assets/130530985/097b5e9b-213d-4f9b-a421-1fadabccc242)

## Dataset

[Daily Temperature of Major Cities](https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities) (Kaggle), `city_temperature.csv`: 2,906,327 daily records from cities worldwide since 1995. This project uses the 27,798 records for Japan (`AvgTemperature` in °F).

## Approach

1. Filter Japan, remove outliers with the IQR rule (this also removes the `-99` missing-value code), and build a `Date` column.
2. Split in time order: 17,726 train, 4,432 validation, 5,540 test values.
3. Scale with `MinMaxScaler`; create windows of 100 days to predict the next day.
4. **Model:** `LSTM(64) → LSTM(128) → LSTM(256) → LSTM(128) → LSTM(64)`, each followed by `Dropout(0.25)`, then `Dense(1)`.
5. **Training:** Huber loss, Adam (lr 0.0001), batch 32, up to 100 epochs; early stopping, checkpoint (`best_model.h5`), `ReduceLROnPlateau`, and a callback that stops once train and validation MAE are below 0.05.
6. **Testing:** predict test windows, invert the scaling, and plot actual vs. predicted.

## Tech Stack

Python, Google Colab, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, seaborn.

## Project Structure

```
DailyTemperature-LSTM/
├── project_timeseries_LSTM.ipynb   # Notebook with outputs
├── project_timeseries_lstm.py      # Script exported from Colab
└── source.txt                      # Dataset and reference links
```

## Getting Started

```bash
git clone https://github.com/harrymardika/DailyTemperature-LSTM.git
cd DailyTemperature-LSTM
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter
```

Download `city_temperature.csv` from Kaggle. The notebook was written for Colab and reads the CSV from Google Drive (`/content/drive/MyDrive/city_temperature.csv`); to run locally, remove the `drive.mount` cell and change the path in `pd.read_csv`.

## Limitations

- Validation windows are built from the training data (`prepare_data(train_scaled, ...)`), so validation MAE is not a true hold-out score.
- A separate scaler is fitted per split, which leaks information from the test set.
- Records from several Japanese cities are modeled as one series.

## References

- https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities
- https://www.kaggle.com/code/yousifbahnasy/lstm-project-new

## Author

**Harry Mardika** · [GitHub](https://github.com/harrymardika)
