import yfinance as yf
import pandas as pd
import numpy as np

def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    
    # Get financial statements
    income_stmt = stock.income_stmt.iloc[:, 0]
    balance_sheet = stock.balance_sheet.iloc[:, 0]
    cashflow = stock.cashflow.iloc[:, 0]
    info = stock.info
    
    return {
        "Return on Equity": info.get("returnOnEquity"),
        "Return on Assets": info.get("returnOnAssets"),
        "Return on Invested Capital": income_stmt.get("EBIT", 0) / (balance_sheet.get("Total Assets", 0) - balance_sheet.get("Total Current Liabilities", 0)),
        "Operating Profit Margin": info.get("operatingMargins"),
        "Net Income Margin": info.get("profitMargins"),
        "Gross Margin": info.get("grossMargins"),
        "Asset Turnover": income_stmt.get("Total Revenue", 0) / balance_sheet.get("Total Assets", 0),
        "Free Cash Flow": cashflow.get("Free Cash Flow", 0),
        "Earnings Stability": info.get("revenueGrowth"),
        "Debt-to-Equity": info.get("debtToEquity"),
        "Interest Coverage": income_stmt.get("EBIT", 0) / income_stmt.get("Interest Expense", 1),
        "Dividend Growth": calculate_dividend_growth(stock.dividends)
    }

def calculate_dividend_growth(dividends):
    if len(dividends) < 8:
        return 0
    current_year = dividends[-4:].sum()
    previous_year = dividends[-8:-4].sum()
    return (current_year - previous_year) / previous_year if previous_year else 0

def rank_stocks(stocks):
    data = {}
    for stock in stocks:
        data[stock] = get_financial_data(stock)
        print(f"Data fetched for {stock}: {data[stock]}")
    
    df = pd.DataFrame(data).T
    
    # Replace infinity values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    # Normalize the data
    for column in df.columns:
        if column != "Free Cash Flow":  # Don't normalize absolute values
            min_val = df[column].min()
            max_val = df[column].max()
            if min_val != max_val:
                df[column] = (df[column] - min_val) / (max_val - min_val)
            else:
                df[column] = 1  # If all values are the same, set to 1
    
    # Calculate the equally weighted score
    df['Score'] = df.drop('Free Cash Flow', axis=1).mean(axis=1)
    
    # Rank the companies based on the score
    df['Rank'] = df['Score'].rank(ascending=False)
    
    return df.sort_values('Rank')

# List of stocks to rank
stocks = [
    "TTD", "IDCC", "CRWD", "ROKU", "ZS", "DOCN", "SYNC.L", "GRAB", "SSYS", "PLTR", "ESTC", "UEC",
    "NVDA", "BAND", "WOOF", "IRDM"
]

rankings = rank_stocks(stocks)

# Reorder columns to put Rank and Score right after the ticker
columns_order = ['Score', 'Rank'] + [col for col in rankings.columns if col not in ['Score', 'Rank']]
rankings = rankings[columns_order]

print("\nFinal ranking table:")
print(rankings)

# Export to CSV with Rank and Score right after the ticker
rankings.to_csv('stock_rankings.csv')
print("\nRanking table exported to 'stock_rankings.csv'")