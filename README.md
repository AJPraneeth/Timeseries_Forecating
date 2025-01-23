# Time Series Forecasting with Yahoo Stock Price Data

## Table of Contents

1. [Dataset](#Dataset)
2. [Model](#Model)
3. [Model Architecture](#Model-Architecture)
4. [Training](#Training)
5. [Evaluation](#Evaluation)

## Dataset

This section provides information about the training dataset.
[Dataset][datasetURL]

[datasetURL]: https://www.kaggle.com/datasets/arashnic/time-series-forecasting-with-yahoo-stock-price/data

## Model

This section provides an overview of the model used for time series forecasting.

## Model Architecture

The model is an LSTM-based neural network with the following architecture:
- Multiple LSTM layers with hidden sizes: [64, 32, 16, 8]
- Fully connected layers for output
- Batch normalization for stability

### LSTM

The LSTM model is implemented with the following components:
- Input gate weights
- Forget gate weights
- Output gate weights
- Cell gate weights

### LSTM_Yahoo_Stock_Price

The LSTM_Yahoo_Stock_Price model is an extension of the LSTM model with additional components:
- Multiple LSTMCell layers
- Fully connected layers for output
- Batch normalization for stability

## Training

The model is trained using the following configuration:
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam with a learning rate of 0.001
- Number of epochs: 50
- Batch size: 5

## Evaluation

The model is evaluated using the following metrics:
- Root Mean Squared Error (RMSE)
- R² Score

The evaluation results include RMSE and R² scores for each feature over two prediction days.
