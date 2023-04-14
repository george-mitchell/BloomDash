# BloomDash
An interactive dash app for financial analysis: 

https://bloom-dash-app.onrender.com

*Note that the app may take a little time to render...*

## Features 

The dash app enables the user to:
* Build a portfolio by choosing up to five stocks and how many shares of each stock.  
* See a donut chart breakdown of their created porfolio. 
* Get the latest treasury rates and closing prices of their chosen stocks. 

Furthermore, the app then allows the user to **analyse** their portfolio: 
* Plotting the normalized closing price of their portfolio alongside the S&P500 index benchmark. 
  * The app then provides the returns % along with the performance multiplier. 
* Plotting the daily returns of the portfolio against the benchmark, and extracing the alpha and beta for the portfolio. 
* Plotting each chosen stock's closing price along with a 10-day and 90-day simple moving average, and also plotting the points where buy or sell signals where generated. 


### Packages

All financial data is obtained using the `yfinance` package for python. 

The app is built using `dash` and `plotly` for python. 
