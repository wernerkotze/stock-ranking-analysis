import { NextResponse } from 'next/server';

export async function GET() {
  // Mock data for demonstration purposes
  const mockData = [
    { Rank: 1, Ticker: "AAPL", "Composite Score": 0.95, "Earnings Yield": 0.05, "Dividend Yield": 0.006, ROE: 0.35, ROIC: 0.30 },
    { Rank: 2, Ticker: "MSFT", "Composite Score": 0.92, "Earnings Yield": 0.04, "Dividend Yield": 0.008, ROE: 0.40, ROIC: 0.35 },
    { Rank: 3, Ticker: "GOOGL", "Composite Score": 0.90, "Earnings Yield": 0.06, "Dividend Yield": 0, ROE: 0.25, ROIC: 0.22 },
    { Rank: 4, Ticker: "AMZN", "Composite Score": 0.88, "Earnings Yield": 0.03, "Dividend Yield": 0, ROE: 0.20, ROIC: 0.18 },
    { Rank: 5, Ticker: "NVDA", "Composite Score": 0.85, "Earnings Yield": 0.02, "Dividend Yield": 0.004, ROE: 0.45, ROIC: 0.40 },
  ];

  return NextResponse.json(mockData);
}

