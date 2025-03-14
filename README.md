# Stock Ranking Analysis Using yFinance

## Overview
This project provides a script that uses the `yfinance` library to fetch financial data for various stocks, calculates several financial metrics, and ranks the stocks based on an equally weighted scoring system. The metrics used for ranking include profitability ratios, efficiency ratios, and growth indicators. The final rankings are then exported to a CSV file for further analysis.

### Key Features
- Fetches financial data for a list of stocks using `yfinance`.
- Computes several financial metrics, including Return on Equity, Operating Profit Margin, Free Cash Flow, and more.
- Normalizes the financial metrics to provide an even basis for comparison.
- Calculates an overall score for each stock and ranks them.
- Outputs the ranked results to a CSV file.

## Requirements
- Python 3.x
- `yfinance` library for fetching stock data
- `pandas` for data manipulation
- `numpy` for numerical operations

### Installation
To get started, install the required packages by running:

```bash
pip install yfinance pandas numpy
```

## Usage
1. **Define the List of Stocks**: Update the `stocks` list with the tickers you want to analyze.
   
   ```python
   stocks = [
       "CRWD", "ROKU", "ZS", "DOCN", "GRAB", "SSYS", "PLTR", "ESTC", "UEC", "NVDA"
   ]
   ```

2. **Run the Script**: Execute the Python script to rank the stocks.
   
   ```bash
   python3 main.py
   ```

3. **Output**: The script will generate a CSV file named `stock_rankings.csv` containing the ranked list of stocks with key financial metrics.

## Code Breakdown
- **`get_financial_data(ticker)`**: This function takes a stock ticker as input and returns a dictionary of financial metrics by fetching data from `yfinance`. It gathers data like Return on Equity, Return on Assets, Operating Margins, and more.

- **`calculate_dividend_growth(dividends)`**: This helper function calculates the year-over-year growth rate of dividends, provided there are enough dividend payouts to make the calculation meaningful.

- **`rank_stocks(stocks)`**: This function iterates through the list of stocks, fetches financial data for each using `get_financial_data()`, and creates a Pandas DataFrame. It normalizes the metrics, calculates a score for each stock, and ranks them accordingly.

### Output CSV
- **Rank and Score**: The CSV output includes the `Rank` and `Score` columns to help identify the best stocks according to the chosen metrics.
- **Metrics Included**: Other financial metrics such as Return on Equity, Debt-to-Equity, Asset Turnover, etc., are also present to provide context for the ranking.

## Normalization
The normalization process ensures that all metrics are on a comparable scale. Each financial metric is normalized to a range between 0 and 1, except for `Free Cash Flow`, which is treated as an absolute value and is not normalized.

## Considerations
- **Missing Data**: In case of missing or unavailable data for a given ticker, the script fills the missing values with zero (`0`). This could affect the ranking accuracy and should be kept in mind.
- **Handling Infinite Values**: If any metrics result in infinite values, these are replaced with `NaN` to avoid skewing the results.
- **Dividend Growth Calculation**: Stocks with fewer than 8 dividend payouts are given a dividend growth value of `0`. This approach helps in maintaining consistency but may need to be adjusted depending on your analysis requirements.

## Improvements and Extensions
- **Error Handling**: The current code lacks comprehensive error handling for missing or incorrect data. Adding `try-except` blocks can make the script more robust.
- **Weighting Metrics**: All metrics are equally weighted, which may not be ideal for every situation. You can modify the script to assign different weights to metrics based on their importance.
- **Sector-Specific Ranking**: Extending the script to rank stocks within the same industry or sector would provide more relevant comparisons.

## How does the ML work?

Alright! Imagine you have a **magic box** 🧙‍♂️ that looks at a bunch of numbers and tries to guess which toys will be the most fun to play with next week.  

### **What This Function Does**  
1️⃣ **Cleans up messy numbers** 🧹  
   - Some numbers are too big or too small, so we fix them so they don’t mess up the magic box.  

2️⃣ **Finds patterns** 🔍  
   - It looks at past toy fun scores (how much kids liked them) and finds hidden clues in the numbers.  

3️⃣ **Learns from history** 📚  
   - The magic box looks at old toy data and figures out which toys became fun and which didn’t.  

4️⃣ **Guesses the future** 🔮  
   - It uses everything it learned to guess which toys will be the best next week.  

5️⃣ **Ranks the toys** 🏆  
   - It makes a list from **most fun** to **least fun**, so you know which toy to pick next!  

### **Why is this cool?**  
- Instead of randomly picking toys (or stocks), the magic box helps you make **smart choices**.  
- It can learn and **get better** over time.  

So basically, this function is like a **smart fortune-teller** for stocks, helping you guess which ones might be great to invest in! 📈💡

## License
This project is licensed under the MIT License.

## Contact
For any questions or improvements, please feel free to reach out or open an issue on the repository.
