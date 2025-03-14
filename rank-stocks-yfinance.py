import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import mstats

def safe_get_latest_column(data, ticker, data_type):
    """Return the latest available column of the DataFrame if it exists, otherwise log and return 0."""
    if data is not None and not data.empty:
        latest_date = data.columns.max()  # Get the most recent date column
        return data[latest_date]
    print(f"Warning: No {data_type} data available for {ticker}")
    return pd.Series(dtype='float64')  # Return empty series if no data is available

def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    
    # Safely fetch financial data
    income_stmt = safe_get_latest_column(stock.income_stmt, ticker, "income statement")
    balance_sheet = safe_get_latest_column(stock.balance_sheet, ticker, "balance sheet")
    cashflow = safe_get_latest_column(stock.cashflow, ticker, "cash flow")
    info = stock.info
    
    # Return relevant metrics
    return {
        "Earnings Yield": income_stmt.get("Net Income", 0) / balance_sheet.get("Total Equity", 1),
        "Dividend Yield": info.get("dividendYield", 0),
        "EPS Growth": info.get("earningsGrowth", 0),
        "Sales Growth": info.get("revenueGrowth", 0),
        "ROE": info.get("returnOnEquity", 0),
        "ROA": info.get("returnOnAssets", 0),
        "ROIC": income_stmt.get("EBIT", 0) / (balance_sheet.get("Total Assets", 1) - balance_sheet.get("Total Current Liabilities", 0)),
        "Gross Margin": info.get("grossMargins", 0),
        "Net Income Margin": info.get("profitMargins", 0),
        "Operating Margin": info.get("operatingMargins", 0),
        "FCF Margin": cashflow.get("Free Cash Flow", 0) / income_stmt.get("Total Revenue", 1),
        "Debt to Equity": balance_sheet.get("Total Debt", 0) / balance_sheet.get("Total Equity", 1),
        "Interest Coverage": income_stmt.get("EBIT", 0) / income_stmt.get("Interest Expense", 1),
        "Price to Sales": info.get("priceToSalesTrailing12Months", 0),
        "Price to Book": info.get("priceToBook", 0)
    }

def winsorize_and_zscore(df, columns):
    """
    Winsorize and Z-score normalize the specified columns in the DataFrame.
    """
    for col in columns:
        if df[col].notnull().any():
            # Winsorize at 5th and 95th percentiles
            df[col] = mstats.winsorize(df[col], limits=[0.05, 0.05])
            # Z-score normalization
            df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    return df

def fetch_price_movements(stocks, periods):
    end_date = pd.Timestamp.today()
    price_movements = pd.DataFrame(index=stocks, columns=periods.keys())

    for stock in stocks:
        try:
            data = yf.download(stock, start=end_date - pd.DateOffset(months=12), end=end_date, progress=False)
            
            # If data has a MultiIndex, rename columns explicitly
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ["Open", "High", "Low", "Close", "Volume"]
            
            # Ensure data is a DataFrame
            if isinstance(data, pd.Series):
                data = data.to_frame().T
            
            print(data)  # For debugging
            
            if not data.empty:
                for period_name, months in periods.items():
                    start_date = end_date - pd.DateOffset(months=months)
                    filtered_data = data.loc[data.index >= start_date]
                    
                    # If filtering results in a Series, handle it accordingly:
                    if isinstance(filtered_data, pd.Series):
                        start_price = filtered_data["Close"]
                    elif not filtered_data.empty:
                        start_price = filtered_data.iloc[0]["Close"]
                    else:
                        print(f"No data for {stock} starting from {start_date}")
                        continue

                    # Get the end price from the last row
                    end_row = data.iloc[-1]
                    if isinstance(end_row, pd.Series):
                        end_price = end_row["Close"]
                    else:
                        end_price = data.iloc[-1]["Close"]

                    price_change = ((end_price - start_price) / start_price) * 100
                    price_movements.loc[stock, period_name] = price_change
        except Exception as e:
            print(f"Error fetching data for {stock}: {e}")

    return price_movements


def rank_stocks(stocks):
    data = {}
    
    # Fetch data for each stock
    for stock in stocks:
        try:
            data[stock] = get_financial_data(stock)
        except Exception as e:
            print(f"Error processing {stock}: {e}")
            continue  # Skip ticker if any error occurs
    
    # Create a DataFrame with absolute values
    df_absolute = pd.DataFrame(data).T
    
    # Replace infinity values with NaN and fill NaNs with 0
    df_absolute.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_absolute.fillna(0, inplace=True)
    
    # Save absolute values to CSV
    df_absolute.to_csv('stock_absolute_values.csv', index=True)
    print("\nAbsolute values table exported to 'stock_absolute_values.csv'")
    
    # List of columns to Winsorize and Z-score
    columns_to_normalize = df_absolute.columns
    
    # Apply Winsorizing and Z-scoring
    df_normalized = winsorize_and_zscore(df_absolute.copy(), columns_to_normalize)
    
    # Calculate composite score (average of all factor Z-scores)
    df_normalized['Composite Score'] = df_normalized.mean(axis=1)
    
    # Rank stocks based on the composite score
    df_normalized['Rank'] = df_normalized['Composite Score'].rank(ascending=False)
    
    # Reorder columns to put Rank and Composite Score first
    columns_order = ['Rank', 'Composite Score'] + [col for col in df_normalized.columns if col not in ['Rank', 'Composite Score']]
    df_normalized = df_normalized[columns_order]
    
    # Save rankings with normalized values to CSV
    df_normalized.to_csv('stock_rankings.csv', index=True)
    print("\nRanking table exported to 'stock_rankings.csv'")
    
    # Fetch price movements
    periods = {"1 Month": 1, "3 Months": 3, "6 Months": 6, "9 Months": 9, "12 Months": 12}
    df_price_movements = fetch_price_movements(stocks, periods)
    
    # Save price movements to CSV
    df_price_movements.to_csv('stock_price_movements.csv', index=True)
    print("\nPrice movements exported to 'stock_price_movements.csv'")
    
    # Combine financial and price movement data
    df_combined = df_normalized.join(df_price_movements)
    
    # Save combined rankings to CSV
    df_combined.to_csv('combined_rankings.csv', index=True)
    print("\nCombined rankings exported to 'combined_rankings.csv'")
    
    return df_combined

# List of stocks to rank
stocks = [
    "AAPL", "GOOG", "AMZN"
]

rankings = rank_stocks(stocks)