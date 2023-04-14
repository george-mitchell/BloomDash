## Portfolio Dashboard

########################################################################

## standard imports
import pandas as pd
import yfinance as yf
import yahoo_fin.stock_info as si
from datetime import date, timedelta
from pandas_datareader.data import DataReader as dr
from statsmodels import regression
import statsmodels.api as sm

## plotting imports
import plotly.graph_objects as go
import plotly.express as px

## Dashboard imports
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

##########################################################################

## Get a list of tickers for the S&P500
tickers = si.tickers_sp500()


## Set the button style
dropdown_style = {
    'background-color': "#6f8fac",
    'color': 'black',
    'height': '40px',
    'width': '200px',
    'font-size': '15px',
    'font-color': '#4EB1CB'
}

input_style = {
    'background-color': "rgb(71, 95, 112)",
    'color': '#4EB1CB',
    'height': '40px',
    'width': '200px'
}




############################### Creating the app ##########################
app = Dash(__name__)
server = app.server

## init portfolio
portfolio = []


## Setting the layout
app.layout = html.Div(
    children= [
    ## make the header
        html.Div(
            className="header",
            children = [
                html.H2("BLOOMDASH"),
                html.H5("A dashboard for building, viewing and analyzing a chosen portfolio.")
            ]
        ),
        html.Hr(),
        ## make the next row of three columns: Stock, Number, Pie Chart
        html.Div(className='row',
            children = [
                ## Left Column: Stock
                html.Div(className='leftcolumn',
                         children=[
                            html.Div(className='cardt',
                                    children=[
                                        html.H3("Stocks"),
                                        html.Label("Choose up to 5 stocks:"),
                                        html.Br(),
                                        html.Br(),
                                        dcc.Dropdown(
                                            options=tickers,
                                            ## value='MSFT',
                                            id="stock-1",
                                            clearable=True,
                                            placeholder="e.g. MSFT",
                                            style = dropdown_style
                                        ),
                                        
                                        dcc.Dropdown(
                                            options=tickers,
                                            id="stock-2",
                                            clearable=True,
                                            placeholder="e.g. MSFT",
                                            style = dropdown_style
                                        ),
                                        
                                        dcc.Dropdown(
                                            options=tickers,
                                            id="stock-3",
                                            clearable=True,
                                            placeholder="e.g. MSFT",
                                            style = dropdown_style
                                        ),
                                        
                                        dcc.Dropdown(
                                            options=tickers,
                                            id="stock-4",
                                            clearable=True,
                                            placeholder="e.g. MSFT",
                                            style = dropdown_style
                                        ),
                                        
                                        dcc.Dropdown(
                                            options=tickers,
                                            id="stock-5",
                                            clearable=True,
                                            placeholder="e.g. MSFT",
                                            style = dropdown_style
                                        ),
                                        html.Br()
                                    ])
                        ]),
                ## Middle Column: Number of Shares to Buy        
                html.Div(className='midleftcolumn',
                         children=[
                            html.Div(className='cardt',
                                     children=[
                                        html.H3("Number of Shares"),
                                        html.Label("Number of shares per stock:"),
                                        html.Br(),
                                        html.Br(),
                                        dcc.Input(
                                            id='input-1',
                                            type='number',
                                            value=0,
                                            placeholder='E.g. 10',
                                            style = dropdown_style
                                        ),
                                        html.Br(),
                                        dcc.Input(
                                            id='input-2',
                                            type='number',
                                            value=0,
                                            style = dropdown_style
                                        ),
                                        html.Br(),
                                        dcc.Input(
                                            id='input-3',
                                            type='number',
                                            value=0,
                                            style = dropdown_style
                                        ),
                                        html.Br(),
                                        dcc.Input(
                                            id='input-4',
                                            type='number',
                                            value=0,
                                            style = dropdown_style
                                        ),
                                        html.Br(),
                                        dcc.Input(
                                            id='input-5',
                                            type='number',
                                            value=0,
                                            style = dropdown_style
                                        ),
                                        html.Br(),
                                        html.Br(),
                                        html.Button(id="submit-stocks", n_clicks=0, children="Build",
                                                    style={"align" : 'right'})
                                     ])
                         ]),
                ## mid Right Column: Donut Chart
                html.Div(className='midrightcolumn',
                         children=[
                            html.Div(className='cardt',
                                     children=[
                                        dcc.Graph(id="pie-chart")
                                     ])
                         ]),
                ## right column
                html.Div(className='rightcolumn',
                         children=[
                            html.Div(className='cardt2',
                                     children=[
                                        html.H3("Portfolio Info"),
                                        html.H5(id="date"),
                                        html.H5("Treasury Rates:",
                                                id='rates'),
                                        html.P(id="port-info")
                                     ])
                         ])
            ]
        ),
######################################### BOTTOM HALF ####################################################3
        html.Hr(),
        html.Div(className='row',
            children=[
                html.Div(className='lcolumn',
                         children=[
                            html.Div(className='cardb',
                                     children=[
                                        html.H3("Analyze Portfolio"),
                                        html.Div(className= "row",
                                                 children = [
                                                    html.Div(
                                                        children = [
                                                            html.Label("Start Date: "),
                                                            dcc.Input(
                                                                id='start-date',
                                                                type='text',
                                                                placeholder='2016-01-01'
                                                            ),
                                                            html.Label(" End Date: "),
                                                            dcc.Input(
                                                                    id='end-date',
                                                                    type='text',
                                                                    placeholder='2023-01-01'
                                                            ),
                                                            html.Button(id="plot-stocks", n_clicks=0, children="Plot",
                                                                        style= {"margin-left" : "10px"}
                                                            ),
                                                        ], style = {"width" : "60%", "float" : "left"}
                                                    ),
                                                    html.Div(
                                                        children = [
                                                            dcc.RadioItems(
                                                                ['Normalized Price', 'Returns', 'Moving Average'],
                                                                id='graph-type',
                                                                ## inline=True,
                                                                value='Normalized Price',
                                                            ),
                                                        ], style = {"width" : "40%", "float" : "left"}
                                                    ),     
                                                 ]),
                                        html.Br(),
                                        dcc.Graph('line-plot',
                                                  style = {"margin-top" : "10px"}),
                                        dcc.Dropdown(
                                                        id="stock-chooser",
                                                        clearable = False,
                                                        style = {
                                                            'background-color': "#6f8fac",
                                                            'color': 'black',
                                                            'height': '40px',
                                                            'width': '80px',
                                                            'font-size': '15px',
                                                            'font-color': '#4EB1CB',
                                                        }
                                                    )   
                                     ])
                         ]),
                html.Div(className='rcolumn',
                         children=[
                            html.Div(className='cardb2',
                                     children=[
                                        html.H3("Results"),
                                        html.Div(id='analysis-output'),
                                     ])
                         ])
            ])




    ]
)




####### Callbacks ##########



#### Dropdown for moving averages
@app.callback(
    Output("stock-chooser", "options"),
    Input("graph-type", "value"),
    State("stock-1", "value"),
    State("stock-2", "value"),
    State("stock-3", "value"),
    State("stock-4", "value"),
    State("stock-5", "value"),
)
def stock_chooser(graph_type,
                  stock_1, stock_2, stock_3, stock_4, stock_5):
    
    ## remove any empty inputs
    stocks = [stock_1, stock_2, stock_3, stock_4, stock_5]
    stocks = [x for x in stocks if x != None]

    if graph_type == "Moving Average":
        return stocks
    else:
        return ["None"]


@app.callback(
    Output('stock-chooser', 'value'),
    Input('stock-chooser', 'options'))
def set_stock_chooser_value(available_options):
    return available_options[0]


##### Pie Chart ######
@app.callback(
    Output("pie-chart", "figure"),
    Output('date', 'children'),
    Output("port-info", "children"),
    Output("rates", "children"),
    Input("submit-stocks", "n_clicks"),
    State("stock-1", "value"),
    State("stock-2", "value"),
    State("stock-3", "value"),
    State("stock-4", "value"),
    State("stock-5", "value"),
    State("input-1", "value"),
    State("input-2", "value"),
    State("input-3", "value"),
    State("input-4", "value"),
    State("input-5", "value"),
)
def generate_pie(n_clicks,
                   stock1,stock2,stock3,stock4,stock5,
                   input1,input2,input3,input4,input5):

    stocks = [stock1,stock2,stock3,stock4,stock5]
    inputs = [input1,input2,input3,input4,input5]

    ## Remove any empty inputs
    stocks = [x for x in stocks if x != None]
    inputs = [x for x in inputs if x != 0]

    ## make the date
    today = date.today()
    prev_date = today - timedelta(days=2)
    start_date = prev_date.strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    date_output = f"Today: {end_date}"

    ## get the treasury rates
    syms = ['DGS10', 'DGS5', 'DGS2', 'DGS1MO', 'DGS3MO']
    yc = dr(syms, 'fred')
    names = dict(zip(syms, ['10yr', '5yr', '2yr', '1m', '3m']))
    yc = yc.rename(columns=names)
    yc = yc[['1m', '3m', '2yr', '5yr', '10yr']]
    yc = yc.iloc[-1,:]

    df = yc.to_frame().T

    rates_table = html.Table(
        children = [
            html.Tbody(
                children=[
                    html.Tr( children = [
                        html.Td("1m"),
                        html.Td(" "),
                        html.Td("3m"),
                        html.Td(" "),
                        html.Td("2yr"),
                        html.Td(" "),
                        html.Td("5yr"),
                        html.Td(" "),
                        html.Td("10yr")
                    ]
                    ),
                    html.Tr( children = [
                        html.Td(df["1m"][0]),
                        html.Td(" "),
                        html.Td(df["3m"][0]),
                        html.Td(" "),
                        html.Td(df["2yr"][0]),
                        html.Td(" "),
                        html.Td(df["5yr"][0]),
                        html.Td(" "),
                        html.Td(df["10yr"][0]),
                    ]
                    )
                ]
            )
        ]
    )

    rates_output = ["Treasury Rates: ", rates_table]

    # rates_output = []
    # for name in df.columns:
    #     rate = df[name][0]
    #     rates_output.append(html.P(f"{name}: {rate}"))
    

    if n_clicks != 0:

        data = yf.download(stocks, start=start_date, end=end_date).Close

        ## get the current closing prices for the stocks
        info_output = []
        if len(stocks) == 1:
            price = round(data[-1], 2)
            tick = stocks[0]
            info_output.append(f"{tick} Current Price: {price}")
        else:
            for tick in stocks:
                price = round(data[tick][-1], 2)
                info_output.append(html.P(f"{tick} Current Price: {price}"))
    
    else:
        stocks = []
        inputs = []
        info_output= ""


    ## Make Pie Chart
    df = pd.DataFrame(
        {
            "Stock": stocks,
            "Number of Shares": inputs,
        }
    )

    ## Make Pie Chart
    fig = px.pie(
        data_frame=df,
        values="Number of Shares",
        names="Stock",
        hole=0.3,
        title="Portfolio Summary",
    )

    fig.update_layout(
        title={
            "x": 0.5,
            "y": 0.95,
            "font": {"size": 20, 'color' : "#4EB1CB"},
            "xanchor": "center",
            "yanchor": "middle",
        },
        legend=dict(
                x=1,
                y=1.02,
                font=dict(
                    family='Latin Modern Sans',
                    size=16,
                    color="#4EB1CB"
                ),
        ),
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        height = 380,
    )
    
    
    
    return fig, date_output, info_output, rates_output




## Lineplot
@app.callback(
    Output("line-plot", "figure"),
    Output("analysis-output", "children"),
    Input("plot-stocks", "n_clicks"),
    Input("stock-chooser", "value"),
    State("start-date", "value"),
    State("end-date", "value"),
    State("graph-type", "value"),
    State("stock-1", "value"),
    State("stock-2", "value"),
    State("stock-3", "value"),
    State("stock-4", "value"),
    State("stock-5", "value"),
    State("input-1", "value"),
    State("input-2", "value"),
    State("input-3", "value"),
    State("input-4", "value"),
    State("input-5", "value"),
)
def generate_line(n_clicks,
                  chosen_stock,
                  start_date,
                  end_date,
                  graph_type,
                  stock1,stock2,stock3,stock4,stock5,
                  input1,input2,input3,input4,input5):
    
    stocks = [stock1,stock2,stock3,stock4,stock5]
    inputs = [input1,input2,input3,input4,input5]

    ## Remove any empty inputs
    stocks = [x for x in stocks if x != None]
    inputs = [x for x in inputs if x != 0]
    
    if len(stocks) != 0:

        ## download index date
        ## make the sp500 index data
        sp500 = yf.download("^GSPC", start=start_date, end=end_date).Close
        norm_sp500 = sp500.div(sp500.iloc[0])*100
        norm_sp500.rename("SP500", inplace=True)

        ## Download portfolio data
        df = yf.download(stocks, start=start_date, end=end_date).Close

        if len(stocks) == 1:
            df = df*inputs[0]
            port_df = df
        else:
            for i in range(len(stocks)):
                df.iloc[:,i] = df.iloc[:,i]*inputs[i]
            port_df = df.sum(axis=1)


        ## Rename the data and normalize
        port_df.rename("Portfolio", inplace=True)
        norm_port_df = port_df.div(port_df.iloc[0])*100

        ## create completed dataframe for use in lineplot
        data = pd.concat([norm_port_df, norm_sp500], axis=1)

        
    
    ## if no stocks in list, generate an empty placeholder plot
    else:
        data = {
            "Portfolio" : [],
            "SP500": []
        }

        fig = px.line(
            data_frame=data,
            y=["Portfolio", "SP500"],
            title="",
        )

        ## remove gridlines from lineplot
        fig.update_xaxes(showgrid=False,
                        color="#4EB1CB",
                        visible = False)
        fig.update_yaxes(showgrid=False,
                        color="#4EB1CB",
                        visible = False)
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
        )

        analysis = f""

        return fig, analysis


    ## Normalized Price Plot
    if graph_type == "Normalized Price":

        fig = px.line(
            data_frame=data,
            y=["Portfolio", "SP500"],
            title="Portfolio and Benchmark Closing Prices",
        )

        ## remove gridlines from lineplot
        fig.update_xaxes(showgrid=False,
                        color="#4EB1CB",
                        )
        fig.update_yaxes(showgrid=False,
                        color="#4EB1CB",
                        tick0 = 0,
                        dtick = 25,
                        nticks = 10)
        
        fig.update_layout(
            title={
                "x": 0.5,
                "y": 0.95,
                "font": {"size": 20, 'color' : "#4EB1CB"},
                "xanchor": "center",
                "yanchor": "middle",
            },
            xaxis_title = "Date",
            yaxis_title = "Normalized Closing Price",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x",
            legend_title_text='',
            height=500,
            legend=dict(
                x=1,
                y=1.02,
                font=dict(
                    family='Latin Modern Sans',
                    size=16,
                    color="#4EB1CB"
                ),
                orientation="h",
                entrywidth=70,
                yanchor="bottom",
                xanchor="right",
            )
        )

        ## analyse the data
        def percent_return(data, index):
            # Calculate percentage return
            init = data.iloc[0,index]
            last = data.iloc[-1,index]
            per_return = (last - init) / init * 100
            return per_return
        
        ## benchmark
        bench_returns = percent_return(data=data, index=1)
        port_returns = percent_return(data=data, index=0)
        multiplier = round(port_returns/bench_returns, 1)

        analysis = [
            html.P(f"From {start_date} to {end_date}:" ),
            html.P(f"S&P 500 would have returned {bench_returns: .2f}%."),
            html.P(f"Your Porfolio would have returned {port_returns: .2f}%"),
            html.P(f"Your Portfolio has a return multipler of {multiplier}x against the benchmark." )
        ]
    

    elif graph_type == "Moving Average":
        
        if len(stocks) == 1:
            new_df = df.copy().to_frame()
        else:
            new_df = df[chosen_stock].copy().to_frame()


        # Calculate 10-day SMA
        new_df['SMA_10'] = new_df[chosen_stock].rolling(window=10).mean()

        # Calculate 30-day SMA
        new_df['SMA_90'] = new_df[chosen_stock].rolling(window=90).mean()

        new_df.dropna(inplace=True)

        # buy signal when shorter term crosses above the longer term
        # buy signal when shorter term crosses above the longer term
        buy_signals = [ x for x in range(0,len(new_df.index)) 
            if new_df.iloc[x,1] >= new_df.iloc[x,2]
            and new_df.iloc[x-1,1] < new_df.iloc[x-1,2]
        ]

        ## sell signal when shorter term crosses above the longer term
        sell_signals = [ x for x in range(0,len(new_df.index)) 
            if new_df.iloc[x,1] <= new_df.iloc[x,2]
            and new_df.iloc[x-1,1] > new_df.iloc[x-1,2]
        ]

        num_buys = len(buy_signals)
        num_sell = len(sell_signals)
        
        buy_signal_df = new_df.iloc[buy_signals, :].copy()
        sell_signal_df = new_df.iloc[sell_signals, :].copy()


        fig = px.line(
            data_frame=new_df,
            y = [chosen_stock, "SMA_10", "SMA_90"],
            title = "10-Day and 90-Day Moving Averages"
        )

        ## Add buy signal points
        fig.add_trace(
            go.Scatter(x = buy_signal_df.index,
                    y = buy_signal_df[chosen_stock],
                    mode="markers",
                    name = "Buy",
                    hoverinfo= "all",
                    marker=dict(size=12, symbol="square", color="green", line=dict(width=1.5, color="green"))
                
                    
            )
        )

        ## add sell signal points
        fig.add_trace(
            go.Scatter(x = sell_signal_df.index,
                    y = sell_signal_df[chosen_stock],
                    mode="markers",
                    name = "Sell",
                    hoverinfo= "all",
                    marker=dict(size=12, symbol="circle", color="red", line=dict(width=1.5, color="red"))
            )
        )

        ## remove gridlines from lineplot
        fig.update_xaxes(showgrid=False,
                        color="#4EB1CB"
        )
        fig.update_yaxes(showgrid=False,
                        color="#4EB1CB",
                        ## tick0 = 0,
                        ## dtick = 25,
                        ## nticks = 10
        )
        
        fig.update_layout(
            title={
                "x": 0.5,
                "y": 0.95,
                "font": {"size": 20, 'color' : "#4EB1CB"},
                "xanchor": "center",
                "yanchor": "middle",
            },
            xaxis_title = "Date",
            yaxis_title = "Closing Price",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x",
            legend_title_text='',
            height=500,
            legend=dict(
                x=1,
                y=1.02,
                font=dict(
                    family='Latin Modern Sans',
                    size=16,
                    color="#4EB1CB"
                ),
                orientation="h",
                entrywidth=70,
                yanchor="bottom",
                xanchor="right",
            )
        )

        analysis = [
            html.P(f"Simple Moving Averages (SMA) can be used to generate Buy/Sell signals for a stock."),
            html.P(f"When the 10-day SMA crosses above the 90-day SMA, we get a BUY signal. \
                   When the 10-day passes below the 90-day we get a SELL signal."),
            html.P(f"This stock had a total of {num_buys} Buy signals and {num_sell} Sell signals."),
            html.P(f"The BUY signals are denoted by a square and the SELL signals are denoted by a circle.")
        ]

        

        ## sell signal when shorter term crosses below the longer term

    elif graph_type == "Returns":
        
        benchmark = sp500.copy().to_frame()
        asset = port_df.copy().to_frame()

        benchmark_returns = benchmark.pct_change().dropna()
        benchmark_returns.rename(columns={"Close" : "Benchmark Returns"}, inplace = True)
        asset_returns = asset.pct_change().dropna()
        asset_returns.rename(columns={ "Portfolio" : "Portfolio Returns"}, inplace=True)

        data = pd.concat([benchmark_returns, asset_returns], axis=1)

        X = benchmark_returns.values
        y = asset_returns.values

        def linreg(x,y) :
            #add a column of 1s to fit alpha
            x = sm.add_constant(x)
            model = regression.linear_model.OLS(y,x).fit()
            #remove the constant now that we're done
            x = x[: , 1]
            return model.params[0], model.params[1]

        alpha, beta = linreg(X,y)

        sharpe_ratio = asset_returns.mean()/asset_returns.std()

        annualized_sharpe = sharpe_ratio * (252 ** 0.5)

        fig = px.scatter(
            data_frame=data,
            x = "Benchmark Returns",
            y = "Portfolio Returns",
            title = "Benchmark Returns vs Portfolio Returns",
            trendline="ols",
            trendline_color_override="green"
        )

        fig.update_xaxes(showgrid=False,
                        color="#4EB1CB"
        )
        fig.update_yaxes(showgrid=False,
                        color="#4EB1CB",
                        ## tick0 = 0,
                        ## dtick = 25,
                        ## nticks = 10
        )
        
        fig.update_layout(
            title={
                "x": 0.5,
                "y": 0.95,
                "font": {"size": 20, 'color' : "#4EB1CB"},
                "xanchor": "center",
                "yanchor": "middle",
            },
            xaxis_title = "Benchmark Returns (Daily %)",
            yaxis_title = "Portfolio Returns (Daily %)",
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x",
            legend_title_text='',
            height=500,
            legend=dict(
                x=1,
                y=1.02,
                font=dict(
                    family='Latin Modern Sans',
                    size=16,
                    color="#4EB1CB"
                ),
                orientation="h",
                entrywidth=70,
                yanchor="bottom",
                xanchor="right",
            )
        )

        analysis = [
            html.P(f"The alpha of a portfolio is a measure of how much the portfolio returns above the benchmark. \
                   The alpha of your portfolio is {alpha}"),
            html.P(f" The beta of a portfolio is a measure of how correlated the portfolio returns are with the benchmark returns. \
                   The beta of your portfolio is {beta}."),
            html.P(f"The Sharpe ratio is a risk-adjusted measure of a portfolio's performance. \
                   The annualized Sharpe Ratio for your porfolio is {annualized_sharpe[0]}.")
        ]
    
    
    return fig, analysis



















## Runs the app
## Note you must change the port from the earlier port value
if __name__ == "__main__":
    app.run(debug=True, port=8010)
