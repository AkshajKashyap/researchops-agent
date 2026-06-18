# Time-Series Forecasting Note

This note describes a local forecasting experiment for hourly energy demand. The task is
one-step-ahead time-series forecasting using the EnergyDemand-2024 dataset, which contains
timestamped load, temperature, and calendar features.

We compare Ridge, LSTM, and Transformer models against a seasonal naive baseline. Models are
evaluated with RMSE and MAE on a chronological split where the final four weeks are held out
for validation.

Results show that the LSTM achieves lower RMSE than Ridge, while the Transformer achieves the
lowest MAE on the validation window. The Ridge model remains useful as a fast baseline.

A reproducibility risk is that random seeds and hardware details are not fully specified in
the retrieved note, so exact neural model results may vary.
