import os
import time
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from scipy.stats import zscore
from scipy.stats.mstats import winsorize as mstats_winsorize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv

# Set your free Alpha Vantage API key here
# Load environment variables from the .env file
load_dotenv()

# Access the variables
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
BASE_URL = "https://www.alphavantage.co/query"

# ------------------------------
# Helper: Safe API Call with Rate Limit Handling
# ------------------------------
def safe_api_call(params):
    """
    Make an API call with given parameters.
    If the response indicates that the free call limit has been exceeded,
    wait 60 seconds and retry.
    """
    while True:
        response = requests.get(BASE_URL, params=params)
        if response.status_code != 200:
            print("HTTP error:", response.status_code)
            time.sleep(10)
            continue
        data = response.json()
        # Alpha Vantage returns a "Note" when call frequency is exceeded.
        if "Note" in data:
            print("API call limit reached. Waiting 60 seconds...")
            time.sleep(60)
            continue
        return data

# ------------------------------
# Fetch Fundamental Data via OVERVIEW endpoint
# ------------------------------
def get_overview(symbol):
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    data = safe_api_call(params)
    if not data or "Symbol" not in data:
        print(f"Error: No valid overview data for {symbol}")
        return {}
    return data

# ------------------------------
# Fetch Historical Price Data using TIME_SERIES_DAILY
# ------------------------------
def get_daily(symbol):
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    data = safe_api_call(params)
    ts_key = "Time Series (Daily)"
    if ts_key not in data:
        print("Error: Unexpected data format for historical prices:", data)
        return None
    df = pd.DataFrame.from_dict(data[ts_key], orient='index')
    # Convert index to datetime and all columns to numeric
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.sort_index(inplace=True)
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

    # Compute a few sample ratios.
    # For example, Earnings Yield as 1 / PERatio if available.
    peratio = to_float(overview.get("PERatio", 0))
    earnings_yield = 1 / peratio if peratio > 0 else 0

    # Gross Margin computed as GrossProfitTTM / RevenueTTM if both exist.
    revenue = to_float(overview.get("RevenueTTM", 0))
    gross_profit = to_float(overview.get("GrossProfitTTM", 0))
    gross_margin = (gross_profit / revenue) if revenue != 0 else 0

    return {
        "Earnings Yield": earnings_yield,
        "Dividend Yield": to_float(overview.get("DividendYield", 0)),
        "EPS Growth": to_float(overview.get("QuarterlyEarningsGrowthYOY", 0)),
        "Sales Growth": to_float(overview.get("QuarterlyRevenueGrowthYOY", 0)),
        "ROE": to_float(overview.get("ReturnOnEquityTTM", 0)),
        "ROA": to_float(overview.get("ReturnOnAssetsTTM", 0)),
        "Gross Margin": gross_margin,
        "Net Income Margin": to_float(overview.get("ProfitMargin", 0)),
        "Operating Margin": to_float(overview.get("OperatingMarginTTM", 0)),
        "Price to Sales": to_float(overview.get("PriceToSalesRatio", 0)),
        "Price to Book": to_float(overview.get("PriceToBookRatio", 0)),
        "MarketCap": to_float(overview.get("MarketCapitalization", 0))
    }

# ------------------------------
# Compute Additional Features from Historical Prices
# ------------------------------
def get_additional_features(symbol, lookback_years=10):
    df = get_daily(symbol)
    if df is None or df.empty:
        print(f"No historical price data for {symbol}")
        return {}
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * lookback_years)
    df = df.loc[start_date:end_date]
    
    # For momentum and return calculations we use the raw "4. close" field.
    close = df["4. close"]
    features = {}
    
    # Momentum factors (assuming trading days; 252 ~ 1 year)
    if len(close) >= 252:
        features["momentum_12m"] = (close.iloc[-1] / close.iloc[-252]) - 1
    else:
        features["momentum_12m"] = 0

    if len(close) >= 126:
        features["momentum_6m"] = (close.iloc[-1] / close.iloc[-126]) - 1
    else:
        features["momentum_6m"] = 0

    if len(close) >= 63:
        features["momentum_3m"] = (close.iloc[-1] / close.iloc[-63]) - 1
    else:
        features["momentum_3m"] = 0

    # Future 1-month return (approx. 22 trading days)
    if len(close) >= 22:
        features["future_return"] = (close.iloc[-1] / close.iloc[-22]) - 1
    else:
        features["future_return"] = 0

    # Volatility (annualized standard deviation of daily returns)
    returns = close.pct_change().dropna()
    features["volatility"] = returns.std() * np.sqrt(252) if not returns.empty else 0

    # Beta and idiosyncratic volatility are not available from free endpoints.
    features["beta"] = 1.0
    features["idiosyncratic_vol"] = features["volatility"]

    # Size factor (Market Cap) we already have in fundamentals; you may merge later.
    # Add a simple feature: average daily volume (if available)
    if "5. volume" in df.columns:
        features["avg_volume"] = df["5. volume"].astype(float).rolling(window=30).mean().iloc[-1]
    else:
        features["avg_volume"] = 0

    return features

# ------------------------------
# Combine Fundamental and Additional Features
# ------------------------------
def combine_features(symbol):
    fundamentals = get_financial_data(symbol)
    additional = get_additional_features(symbol)
    # Merge the two dictionaries.
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
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

# ------------------------------
# ML Model & Ranking using XGBoost
# ------------------------------
def run_ml_model(df, feature_cols):
    for col in feature_cols:
        # Winsorize and compute z-scores.
        df[col] = mstats_winsorize(df[col].fillna(df[col].median()), limits=[0.05, 0.05])
        # For certain valuation metrics lower is better so invert the z-score.
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
    
    # Build combined feature DataFrame
    df_features = build_feature_dataframe(symbols)
    print("Combined Features DataFrame:")
    print(df_features)
    
    # Save features to CSV if desired
    df_features.to_csv("combined_stock_features.csv", index=True)
    print("Combined stock features saved to 'combined_stock_features.csv'")
    
    # Define feature columns (all except the target 'future_return')
    feature_cols = [col for col in df_features.columns if col != 'future_return']
    ranked_df, model = run_ml_model(df_features, feature_cols)
    
    ranked_df.to_csv("ml_combined_stock_rankings.csv", index=True)
    print("ML-based stock rankings saved to 'ml_combined_stock_rankings.csv'")
