# 📈 Tesla Stock Price Prediction with LSTM

> **A stacked LSTM neural network that forecasts Tesla's daily closing price from 10-day rolling windows of price, volume, and engineered time features.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)](https://www.tensorflow.org/)
[![LSTM](https://img.shields.io/badge/Architecture-Stacked%20LSTM-8B5CF6)](https://en.wikipedia.org/wiki/Long_short-term_memory)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Why LSTM for Stock Data](#why-lstm-for-stock-data)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Step 1: Imports and Data Loading](#step-1-imports-and-data-loading)
  - [Step 2: Temporal Feature Engineering](#step-2-temporal-feature-engineering)
  - [Step 3: Scaling with MinMaxScaler](#step-3-scaling-with-minmaxscaler)
  - [Step 4: Sequence Construction](#step-4-sequence-construction)
  - [Step 5: Train/Test Split and Model Build](#step-5-traintest-split-and-model-build)
  - [Step 6: Training with Early Stopping](#step-6-training-with-early-stopping)
  - [Step 7: Training Curve Visualisation](#step-7-training-curve-visualisation)
  - [Step 8: Predictions and Metrics](#step-8-predictions-and-metrics)
  - [Step 9: Result Visualisations](#step-9-result-visualisations)
- [Key Design Decisions](#key-design-decisions)
- [Results](#results)
- [Important Methodological Notes](#important-methodological-notes)
- [How to Reproduce](#how-to-reproduce)
- [Financial Disclaimer](#financial-disclaimer)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Roadmap](#roadmap)
- [Author](#author)
- [License](#license)

---

## Overview

This project builds a **Recurrent Neural Network (RNN) with stacked LSTM layers** to forecast Tesla's daily closing price using historical OHLCV data and engineered time features. The model looks at the last 10 trading days of data and predicts the next day's closing price.

The full pipeline runs in a single Kaggle notebook, from raw CSV ingestion through feature engineering, scaling, sequence construction, model training, evaluation, and prediction visualisation.

This is a hands-on exploration of **time-series deep learning**, structured as a teaching project with clear step-by-step commentary. It is **not** a production trading system, and the [Financial Disclaimer](#financial-disclaimer) and [Important Methodological Notes](#important-methodological-notes) sections below are required reading for anyone thinking about extending this work.

---

## Why LSTM for Stock Data

Stock prices are sequential: today's price is not independent of yesterday's, last week's, or last quarter's. A simple feedforward network would see each day as an isolated sample and miss temporal patterns entirely.

**Long Short-Term Memory (LSTM)** networks were designed specifically to model sequential dependencies. Each LSTM cell maintains an internal state across time steps and uses three gates (forget, input, output) to decide what information from the past to keep, what to discard, and what to output. For stock data, this means the network can learn patterns like:

- Short-term momentum (the last 3–5 days trending up)
- Volume spikes that precede price movements
- Day-of-week effects (Mondays vs Fridays)
- Longer-term seasonality when combined with engineered time features

LSTM is the natural architectural fit for this problem — more expressive than ARIMA, more temporally aware than a standard feedforward network, and more stable than a vanilla RNN thanks to the gated architecture that mitigates vanishing gradients.

---

## Dataset

**Name:** Tesla Stock Price Data (TSLA)
**Source:** Kaggle (`/kaggle/input/tesla-stock-price-data/TSLA-2.csv`)
**Asset:** Tesla Inc. (NASDAQ: TSLA)
**Frequency:** Daily
**Coverage:** Historical price data covering Tesla's public trading history

### Native Columns

| Column | Description |
|--------|-------------|
| `Date` | Trading date |
| `Open` | Opening price for the day |
| `High` | Highest price during the day |
| `Low` | Lowest price during the day |
| `Close` | **Target variable** — closing price |
| `Volume` | Number of shares traded |
| `Adj Close` | Dividend-adjusted closing price (if present) |

The standard OHLCV schema used by virtually every public equity price dataset.

---

## Feature Engineering

Beyond the raw OHLCV columns, the notebook derives five **temporal features** from the `Date` column:

| Feature | Purpose |
|---------|---------|
| `Year` | Captures long-term regime shifts (pre/post 2020 rally, post-split era) |
| `Month` | Captures annual seasonality (January effect, summer lulls) |
| `Day` | Day of the month — weak signal but low cost to include |
| `DayOfWeek` | Monday through Friday patterns (Monday mean reversion, Friday effects) |
| `Quarter` | Quarterly earnings and reporting seasonality |

These features are extracted once, scaled alongside the price features, and included in every sample in the 10-day rolling window. The model gets to learn whether these temporal markers are useful rather than the engineer having to prove them upfront.

Final feature set passed to the LSTM: **10 features** per time step:

```
[Open, High, Low, Close, Volume, Year, Month, Day, DayOfWeek, Quarter]
```

---

## Model Architecture

A two-layer stacked LSTM with dropout regularisation and a single linear output.

```
Input Shape: (batch, 10 timesteps, 10 features)
│
├── LSTM(64 units, return_sequences=True)
│   └── Returns (batch, 10, 64) — full sequence passed to next layer
│
├── Dropout(0.2)
│
├── LSTM(32 units, return_sequences=False)
│   └── Returns (batch, 32) — only the last time step's hidden state
│
├── Dropout(0.2)
│
└── Dense(1) — scalar predicted Close price (scaled)
```

**Why stacked LSTM?**

A single LSTM layer captures temporal dependencies but is limited in its ability to learn hierarchical representations of those dependencies. Stacking a second LSTM on top lets the network learn:

- **Layer 1 (64 units):** Low-level temporal patterns — short-term momentum, volatility bursts, daily volume-price relationships
- **Layer 2 (32 units):** Higher-order patterns built from the first layer's sequence output — longer-term trend shape, reversal signals

The `return_sequences=True` on the first LSTM is what allows the stacking to work. It passes the full 10-step sequence up to the second layer, which then compresses it into a single 32-dim vector for the final prediction.

**Why Dropout 0.2 between LSTMs?**

Dropout regularisation forces the network to not rely too heavily on any single neuron, reducing overfitting. 0.2 (20%) is light but effective for sequence models — heavier dropout on LSTMs can destabilise training because the cell state accumulates information over many time steps.

**Why a single Dense(1) output?**

This is a regression problem with one target (the next-day Close price, scaled). A single linear output neuron is the standard choice. The MinMaxScaler is inverted on the prediction side if you want to read the result back in dollars.

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Sequence length** | 10 days | Two trading weeks — captures short-term momentum without over-smoothing |
| **Batch size** | 64 | Standard for LSTM training, balances gradient quality with training speed |
| **Max epochs** | 100 | Generous ceiling; early stopping halts training when validation plateaus |
| **Optimiser** | Adam (lr=0.001) | Reliable default for RNN training |
| **Loss** | MSE (Mean Squared Error) | Standard regression loss, penalises large errors quadratically |
| **Early stopping** | patience=5 on `val_loss`, `restore_best_weights=True` | Stops training once validation loss stops improving, rolls back to best weights |
| **Train/test split** | 80/20 via `train_test_split` (`random_state=42`) | Standard split ratio — **but see [Methodological Notes](#important-methodological-notes) below** |

### Scaling Strategy

All 10 features are scaled using **`MinMaxScaler(feature_range=(0, 1))`**. This maps every column to the [0, 1] range, which matters for LSTM training because:

- Gradient updates are more stable when feature magnitudes are comparable
- The sigmoid and tanh activations inside LSTM cells operate naturally in this range
- Large Volume values (millions) don't dominate price values (hundreds) during training

The scaler is fit on the training data. In the current notebook it is fit on the full dataset before splitting, which is a methodological compromise discussed in the [Methodological Notes](#important-methodological-notes) section.

---

## Pipeline Walkthrough

The notebook runs nine steps end to end. Below is a detailed walkthrough.

---

### Step 1: Imports and Data Loading

**What it does:** Imports pandas, NumPy, scikit-learn utilities (train/test split, scalers, metrics), Matplotlib, Seaborn, and Keras layers (Dense, LSTM, Dropout). Loads `TSLA-2.csv` into a pandas DataFrame.

---

### Step 2: Temporal Feature Engineering

**What it does:** Converts the `Date` column to pandas datetime, sorts the DataFrame chronologically, and extracts five temporal features (`Year`, `Month`, `Day`, `DayOfWeek`, `Quarter`). Sets `Date` as the index.

The chronological sort is essential — without it, downstream sequence construction would mix days from different eras into single 10-day windows, completely breaking the time-series assumption.

---

### Step 3: Scaling with MinMaxScaler

**What it does:** Copies the DataFrame into `tsla_scaled` and applies `MinMaxScaler(feature_range=(0, 1))` to all ten feature columns: Open, High, Low, Close, Volume, Year, Month, Day, DayOfWeek, Quarter.

After this step, every value in the working DataFrame lies in [0, 1], which is what the LSTM expects.

---

### Step 4: Sequence Construction

**What it does:** Builds the 3D input tensor that LSTM requires. Loops through the scaled DataFrame from index `sequence_length=10` to the end, and for each index:

- `X[i]` is the previous 10 rows × 10 features (a 10×10 matrix)
- `y[i]` is the Close price on row `i` (a scalar)

Final tensor shapes:
- `X.shape = (n_samples - 10, 10, 10)` — `(samples, timesteps, features)`
- `y.shape = (n_samples - 10,)` — the next-day Close for each window

This is the canonical sliding-window construction for LSTM time-series tasks.

---

### Step 5: Train/Test Split and Model Build

**What it does:** Splits `X` and `y` into train and test sets using `train_test_split(test_size=0.2, random_state=42)`. Constructs the two-layer LSTM model with Dropout between layers and a single Dense output neuron.

---

### Step 6: Training with Early Stopping

**What it does:** Compiles the model with Adam optimiser and MSE loss, sets up an `EarlyStopping` callback monitoring `val_loss` with patience 5, and runs `model.fit()` for up to 100 epochs with batch size 64.

Training typically halts around epoch 20–40 once validation loss plateaus. The `restore_best_weights=True` setting ensures the final model uses the weights from the best validation epoch, not the last one.

---

### Step 7: Training Curve Visualisation

**What it does:** Produces a detailed training loss plot using a custom `plot_enhanced_loss` function:

- Training and validation loss curves over all epochs
- Moving averages (window=5) overlaid on both curves for trend visualisation
- Annotations marking the minimum training and validation loss with their epoch numbers
- A summary box showing total epochs, final train loss, final validation loss

This gives an at-a-glance view of whether training converged cleanly, whether the model overfit, and when the best validation performance was reached.

---

### Step 8: Predictions and Metrics

**What it does:** Runs `model.predict(X_test)` to generate predicted (scaled) Close prices, then computes three regression metrics:

- **MSE (Mean Squared Error)** — average squared error
- **MAE (Mean Absolute Error)** — average absolute error in scaled units
- **R² Score** — proportion of variance explained

All metrics are computed on the scaled targets. To get metrics back in dollar terms, the scaler would need to be inverted on both `y_test` and `y_pred`.

---

### Step 9: Result Visualisations

**What it does:** Produces a 2×2 figure with:

1. **Metrics text block** — MSE, MAE, R² values printed cleanly
2. **True vs Predicted scatter** — actual vs predicted values with a red diagonal showing the line of perfect prediction
3. **Error distribution histogram** — `y_test - y_pred` residuals with KDE overlay, checks for systematic bias
4. **True vs Predicted line plot** — both series plotted over the test index to visually compare their shape

A healthy model will show the scatter hugging the diagonal, residuals centred at zero, and the two line plots moving together closely.

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **Sequence length** | 10 days (two trading weeks) | Long enough to capture short-term momentum, short enough to keep training fast and the target variable stable |
| **Architecture** | 2-layer stacked LSTM (64 → 32) | Hierarchical feature learning without making the network so deep it overfits a single-stock dataset |
| **Dropout rate** | 0.2 between LSTMs | Balanced regularisation — heavier dropout destabilises sequence models |
| **Scaling** | MinMaxScaler [0, 1] | Natural range for LSTM gate activations, keeps Volume from dominating gradients |
| **Features** | OHLCV + 5 time features | Price + volume captures market dynamics, time features let the model discover seasonality |
| **Loss function** | MSE | Penalises large errors heavily, appropriate for price prediction where outlier errors cost the most |
| **Best-epoch recovery** | `restore_best_weights=True` | Guarantees the final model is the best the training run ever produced, not just the last |
| **Early stopping patience** | 5 epochs | Gives the model room to briefly plateau and recover without wasting compute on flat training |

---

## Results

Typical metrics on the test set (scaled [0, 1] targets):

| Metric | Value |
|--------|-------|
| **MSE** | ~0.00042 |
| **MAE** | ~0.0105 |
| **R² Score** | ~0.993 |

These numbers look very strong — R² of 0.993 means the model explains approximately 99% of the variance in the scaled Close price on the test set. **But they should be read alongside the methodological notes in the next section, because the headline numbers are inflated by a design choice that is common in teaching notebooks but problematic for honest time-series evaluation.**

---

## Important Methodological Notes

This notebook is designed for learning, not for deployment. Two methodological considerations are worth naming explicitly.

### 1. Random Train/Test Split on Time-Series Data

The notebook uses `train_test_split(X, y, test_size=0.2, random_state=42)` with the default `shuffle=True`. On time-series data, this causes **data leakage**: the model sees windows from late in the dataset during training, and is then evaluated on randomly-scattered windows that are often temporally adjacent to training windows. The model is effectively interpolating within a known range rather than extrapolating forward.

**This is why the reported R² is so high (~0.993).** A genuinely out-of-sample time-series evaluation — train on 2010–2020, test on 2021–2023 — would produce noticeably lower numbers, often in the R² ~0.70–0.85 range on stock data.

**The correct approach for production time-series evaluation:**

```python
# Instead of random split, use temporal split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

Or use `TimeSeriesSplit` from scikit-learn for a rolling-window cross-validation strategy.

### 2. Scaler Fit on Full Dataset

The `MinMaxScaler` is fit on the entire dataset before the split, which means the scaling parameters (min and max per feature) incorporate information from the test set. In a strict out-of-sample evaluation, the scaler should be fit only on training data and applied to test data.

### Why This Matters

If someone were to take this model as-is and try to use it to trade, the live performance would be significantly worse than the test metrics suggest. The gap between "evaluated correctly" and "evaluated with leakage" on time-series data is typically the difference between a usable model and a system that loses money.

This notebook is honest research-and-learning code that illustrates the LSTM pipeline cleanly. The fix for both issues is straightforward, and addressing them is the top item on the [Roadmap](#roadmap).

---

## How to Reproduce

### Option 1: Run on Kaggle (Recommended)

1. Open [Kaggle](https://www.kaggle.com/) and sign in
2. Create a new notebook
3. Attach the Tesla stock dataset from the Kaggle data tab
4. Upload `tesla-stock-prediction.ipynb` or copy the code cells
5. Run all cells top to bottom

Expected runtime: approximately 5–15 minutes on a T4 GPU, 15–25 minutes on CPU.

### Option 2: Run Locally

```bash
# Clone the repo
git clone https://github.com/[your-username]/tesla-stock-prediction.git
cd tesla-stock-prediction

# Install dependencies
pip install -r requirements.txt

# Place TSLA-2.csv in a local data/ directory
# Update the CSV path in the notebook from /kaggle/input/... to data/TSLA-2.csv

# Run the notebook
jupyter notebook tesla-stock-prediction.ipynb
```

### Hardware Recommendations

- **CPU:** Works fine. Training takes roughly 15–25 minutes
- **GPU:** Optional speedup to 5–15 minutes. Not necessary for a dataset this size

---

## Financial Disclaimer

**This project is for educational and research purposes only. It is not financial advice.**

- The model is a technical demonstration of LSTM time-series forecasting, not a trading system
- Past price movements do not guarantee future returns
- The evaluation methodology has documented limitations (see [Methodological Notes](#important-methodological-notes)) that make the reported metrics optimistic
- No transaction costs, slippage, or market impact are modelled
- No risk management, position sizing, or portfolio construction is included
- Using this model — or any model — to make actual investment decisions is at your own risk

If you are thinking about algorithmic trading, start with backtesting frameworks designed for the purpose (such as `backtrader`, `zipline`, or `QuantConnect`), use walk-forward validation, model transaction costs realistically, and understand that most retail algorithmic strategies lose money after costs.

---

## Repository Structure

```
.
├── README.md                           # This file
├── tesla-stock-prediction.ipynb        # Complete training and evaluation notebook
├── requirements.txt                    # Python dependencies
├── data/                               # (local, not committed)
│   └── TSLA-2.csv                      # Tesla stock price data
├── outputs/                            # (generated)
│   ├── model_loss_plot.png
│   └── prediction_analysis.png
└── LICENSE
```

---

## Dependencies

```
tensorflow>=2.15.0
numpy>=1.26.0
pandas>=2.0.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

Install with:

```bash
pip install -r requirements.txt
```

On Kaggle, everything is pre-installed. No setup required.

---

## Roadmap

Improvements that would move this from a teaching notebook to a more rigorous time-series baseline:

- **Fix the time-series split** — use temporal ordering (train on early data, test on later data) instead of random shuffling, eliminating the leakage that inflates current R²
- **Fit scaler on train only** — proper methodology for truly out-of-sample evaluation
- **Walk-forward validation** — use `TimeSeriesSplit` for robust cross-validation
- **Multi-step forecasting** — predict the next 5 or 30 days, not just 1, closer to how the model would actually be used
- **Add more features** — technical indicators (RSI, MACD, Bollinger Bands), macro signals (VIX, interest rates), sector ETF performance (XLK, QQQ)
- **Attention mechanism** — add an attention layer on top of the LSTM stack to let the model weight past days differently
- **Transformer baseline** — compare against a small time-series transformer (e.g., PatchTST, Informer)
- **Ensemble** — combine multiple LSTM models trained with different random seeds and sequence lengths
- **Inverse-transform metrics** — report MSE and MAE in dollar terms for interpretability
- **Backtesting integration** — convert predictions into a trading strategy and evaluate with realistic transaction costs
- **Multi-stock generalisation** — train jointly on TSLA, AAPL, NVDA, etc. to see if the network learns generalisable price patterns or only TSLA-specific ones
- **Volatility forecasting** — extend the model to also predict realised volatility, which is often more useful for trading than price level

---

## Author

**Collins Lemeke**

This project explores the fundamentals of time-series deep learning using LSTM architectures, structured as a teaching and learning exercise. It connects to broader interests in applied machine learning and the careful application of ML methods to real-world problems where methodology matters as much as architecture.

For questions, feedback, or corrections on the methodological notes, open a GitHub issue.

---

## License

MIT License. Free to use, modify, and distribute. See [LICENSE](LICENSE) for full terms.

---

> *Built with TensorFlow, Keras, and stacked LSTM layers. Designed as a teaching notebook for time-series deep learning. Read the Methodological Notes section before using these metrics for anything serious.*
