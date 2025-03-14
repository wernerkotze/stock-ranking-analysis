import yfinance as yf
dat = yf.Ticker("GOOG")
dat.info
print(dat.info)