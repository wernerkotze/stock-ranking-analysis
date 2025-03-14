import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from scipy.stats import zscore
from scipy.stats.mstats import winsorize as mstats_winsorize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ------------------------------
# Fetch Fundamental Data via yfinance info
# ------------------------------
def get_overview(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    if not info or "symbol" not in info:
        print(f"Error: No valid overview data for {symbol}")
        return {}
    return info

# ------------------------------
# Fetch Historical Price Data using yfinance download
# ------------------------------
def get_daily(symbol):
    end_date = datetime.now()
    df = yf.download(symbol, start="1900-01-01", end=end_date, progress=False)
    if df is None or df.empty:
        print(f"Error: No historical price data for {symbol}")
        return None
    df.sort_index(inplace=True)
    # If columns come as a MultiIndex, flatten them:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

# ------------------------------
# Compute Fundamental Features from Overview Data
# ------------------------------
def get_financial_data(symbol):
    overview = get_overview(symbol)
    if not overview:
        return {}
    
    def to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    peratio = to_float(overview.get("trailingPE", 0))
    earnings_yield = 1 / peratio if peratio > 0 else 0
    gross_margin = to_float(overview.get("grossMargins", 0))
    
    return {
        "Earnings Yield": earnings_yield,
        "Dividend Yield": to_float(overview.get("dividendYield", 0)),
        "EPS Growth": to_float(overview.get("earningsGrowth", 0)),
        "Sales Growth": to_float(overview.get("revenueGrowth", 0)),
        "ROE": to_float(overview.get("returnOnEquity", 0)),
        "ROA": to_float(overview.get("returnOnAssets", 0)),
        "Gross Margin": gross_margin,
        "Net Income Margin": to_float(overview.get("profitMargins", 0)),
        "Operating Margin": to_float(overview.get("operatingMargins", 0)),
        "Price to Sales": to_float(overview.get("priceToSalesTrailing12Months", 0)),
        "Price to Book": to_float(overview.get("priceToBook", 0)),
        "MarketCap": to_float(overview.get("marketCap", 0))
    }

# ------------------------------
# Compute Additional Features from Historical Prices
# ------------------------------
def get_additional_features(symbol, lookback_years=10):
    df = get_daily(symbol)
    if df is None or df.empty:
        print(f"No historical price data for {symbol}")
        return {}
    
    # Use only the last "lookback_years" of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * lookback_years)
    df = df.loc[start_date:end_date]
    
    if df.empty or "Close" not in df.columns:
        print(f"Insufficient data for {symbol} in the period {start_date} to {end_date}")
        return {}
    
    close = df["Close"]
    features = {}
    
    # Momentum factors (using approximate trading day counts)
    features["momentum_12m"] = (close.iloc[-1] / close.iloc[-252] - 1) if len(close) >= 252 else 0
    features["momentum_6m"]  = (close.iloc[-1] / close.iloc[-126] - 1) if len(close) >= 126 else 0
    features["momentum_3m"]  = (close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else 0
    # Future return (using last 22 trading days as approximation for 1 month)
    features["future_return"] = (close.iloc[-1] / close.iloc[-22] - 1) if len(close) >= 22 else 0

    # Volatility (annualized standard deviation of daily returns)
    returns = close.pct_change().dropna()
    features["volatility"] = returns.std() * np.sqrt(252) if not returns.empty else 0

    # Beta and idiosyncratic volatility (set defaults as these require market data)
    features["beta"] = 1.0
    features["idiosyncratic_vol"] = features["volatility"]

    # Average daily volume (rolling 30-day average)
    if "Volume" in df.columns:
        features["avg_volume"] = df["Volume"].astype(float).rolling(window=30).mean().iloc[-1]
    else:
        features["avg_volume"] = 0

    return features

# ------------------------------
# Combine Fundamental and Additional Features
# ------------------------------
def combine_features(symbol):
    fundamentals = get_financial_data(symbol)
    additional = get_additional_features(symbol)
    combined = {**fundamentals, **additional}
    return combined

# ------------------------------
# Build a DataFrame from Features for a List of Symbols
# ------------------------------
def build_feature_dataframe(symbols):
    data = {}
    for sym in symbols:
        try:
            features = combine_features(sym)
            if features:
                data[sym] = features
            else:
                print(f"No features available for {sym}")
        except Exception as e:
            print(f"Error processing {sym}: {e}")
    df = pd.DataFrame(data).T
    # Ensure numeric types (convert objects to numeric where possible)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

# ------------------------------
# Fetch Price Movements (Separate from the ML target)
# ------------------------------
def fetch_price_movements(stocks, periods):
    """
    For each stock, download 12 months of historical data and compute
    the percentage change in the "Close" price over various periods.
    """
    end_date = pd.Timestamp.today()
    price_movements = pd.DataFrame(index=stocks, columns=periods.keys())

    for stock in stocks:
        try:
            data = yf.download(stock, start=end_date - pd.DateOffset(months=12), end=end_date, progress=False)
            if data is None or data.empty:
                print(f"No historical data for {stock}")
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] for c in data.columns]
            if isinstance(data, pd.Series):
                data = data.to_frame().T

            for period_name, months in periods.items():
                period_start = end_date - pd.DateOffset(months=months)
                filtered_data = data.loc[data.index >= period_start]
                if filtered_data.empty:
                    print(f"No data for {stock} in period: {period_name}")
                    continue
                start_price = filtered_data.iloc[0]["Close"]
                end_price = filtered_data.iloc[-1]["Close"]
                price_change = ((end_price - start_price) / start_price) * 100
                price_movements.loc[stock, period_name] = price_change
        except Exception as e:
            print(f"Error fetching price movements for {stock}: {e}")
    return price_movements

# ------------------------------
# ML Model & Ranking using XGBoost
# ------------------------------
def run_ml_model(df, feature_cols):
    for col in feature_cols:
        df[col] = mstats_winsorize(df[col].fillna(df[col].median()), limits=[0.05, 0.05])
        if col in ['Price to Sales', 'Price to Book']:
            df[f'{col}_z'] = -zscore(df[col])
        else:
            df[f'{col}_z'] = zscore(df[col])
    z_cols = [f'{col}_z' for col in feature_cols]
    X = df[z_cols]
    y = df['future_return']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100,
                             max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f"Training RMSE: {train_rmse:.4f}")
    
    df['predicted_return'] = model.predict(X)
    df['rank'] = df['predicted_return'].rank(ascending=False, method="first")
    
    return df.sort_values('rank'), model

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    # List of sample symbols (adjust as needed)
    symbols = ["IBM", "AAPL", "MSFT", "GOOGL", "AMZN"]
    
    # Build combined feature DataFrame (fundamentals + additional features)
    df_features = build_feature_dataframe(symbols)
    print("Combined Features DataFrame:")
    print(df_features)
    df_features.to_csv("combined_stock_features.csv", index=True)
    print("Combined stock features saved to 'combined_stock_features.csv'")
    
    # Define feature columns (all except the target 'future_return')
    feature_cols = [col for col in df_features.columns if col != 'future_return']
    ranked_df, model = run_ml_model(df_features, feature_cols)
    ranked_df.to_csv("ml_combined_stock_rankings.csv", index=True)
    print("ML-based stock rankings saved to 'ml_combined_stock_rankings.csv'")
    
    # Fetch additional price movement data (for extra context, not used in ML target)
    periods = {"1 Month": 1, "3 Months": 3, "6 Months": 6, "9 Months": 9, "12 Months": 12}
    df_price_movements = fetch_price_movements(symbols, periods)
    df_price_movements.to_csv("stock_price_movements.csv", index=True)
    print("Price movements saved to 'stock_price_movements.csv'")
    
    # Optionally, join price movement data with ML rankings for a combined view.
    df_combined = ranked_df.join(df_price_movements)
    df_combined.to_csv("combined_rankings.csv", index=True)
    print("Combined rankings exported to 'combined_rankings.csv'")
