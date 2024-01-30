import os
import pandas as pd
from riskalphaindics.main import moving_average, exponential_moving_average, relative_strength_index, average_true_range, moving_average_convergence_divergence

# Load sample data
current_dir = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(os.path.join(current_dir, 'sample.csv'))

def test_moving_average():
    ma = moving_average(data, window=20)
    assert len(ma) == len(data)
    assert ma.isna().sum() < len(data)  # Ensures not all values are NaN

def test_exponential_moving_average():
    ema = exponential_moving_average(data, window=20)
    assert len(ema) == len(data)
    assert ema.isna().sum() < len(data)

def test_relative_strength_index():
    rsi = relative_strength_index(data, periods=14)
    assert len(rsi) == len(data)
    assert rsi.isna().sum() < len(data)
    assert all(0 <= value <= 100 for value in rsi.dropna())  # RSI should be between 0 and 100

def test_average_true_range():
    atr = average_true_range(data, window=14)
    assert len(atr) == len(data)
    assert atr.isna().sum() < len(data)

def test_moving_average_convergence_divergence():
    macd, signal_line = moving_average_convergence_divergence(data, window_slow=26, window_fast=12, signal=9)
    assert len(macd) == len(data)
    assert len(signal_line) == len(data)
    assert macd.isna().sum() < len(data)
    assert signal_line.isna().sum() < len(data)
