import yfinance as yf
from mlp_for_llm_sim import MLP

def fetch_market_indicator(ticker="AAPL", period="1mo", interval="1d"):
    """
    Fetches historical market data for a given ticker using yfinance and computes
    a simple market indicator: the average closing price.

    Args:
        ticker (str): Stock ticker symbol (default "AAPL").
        period (str): Time period for data (default "1mo").
        interval (str): Data interval (default "1d").

    Returns:
        float: The average closing price.
    """
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
    if data.empty:
        raise ValueError("No data fetched. Check ticker, period, or interval.")
    return float(data['Close'].mean())

if __name__ == "__main__":
    ticker = "AAPL"
    indicator = fetch_market_indicator(ticker)
    print(f"Market indicator for {ticker}: {indicator:.2f}")
    
    mlp = MLP()
    analysis = mlp.analyze(int(indicator))
    print(f"MLP analysis output: {analysis}")

"""
Rationale:

I used yfinance to fetch real market data for AAPL and calculated the average closing price 
as a simple market indicator. This indicator is then used as input for the MLP, which simulates 
a large language model's analysis of market conditions. This approach connects real financial data 
to our simulation, showing a basic but realistic integration that could be expanded with more 
sophisticated indicators in the future.
"""
