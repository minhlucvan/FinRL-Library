import pandas as pd
import numpy as np

from pyfolio import timeseries 
import pyfolio

from finrl.marketdata.datadowloader import get_data_downloader
from finrl.config import config

import matplotlib.pyplot as plt
import matplotlib
from backtesting import Backtest, Strategy


DataDowloader = get_data_downloader(config.DATA_PROVIDER)

def BacktestResults(states, strategy):
    bt = Backtest(states, strategy, cash=config.INITIAL_AMMOUNT, commission=config.TRANSACTION_COST_PCT)
    stats = bt.run()
    print(stats)
    return bt

def BackTestStats(account_value):
    df = account_value.copy()
    df=get_daily_return(df)
    DRL_strat = backtest_strat(df)
    perf_func = timeseries.perf_stats 
    perf_stats_all = perf_func( returns=DRL_strat, 
                                factor_returns=DRL_strat, 
                                 positions=None, transactions=None, turnover_denom="AGB")
    print(perf_stats_all)
    return perf_stats_all

def BaselineStats(baseline_ticker = config.BASELINE_TICKER, 
                  baseline_start = config.START_TRADE_DATE, 
                  baseline_end = config.END_DATE):

    bli, dow_strat = baseline_strat(ticker = baseline_ticker, 
                                    start = baseline_start, 
                                    end = baseline_end)
    perf_func = timeseries.perf_stats 
    perf_stats_all = perf_func( returns=dow_strat, 
                                factor_returns=dow_strat, 
                                 positions=None, transactions=None, turnover_denom="AGB")
    print(perf_stats_all)
    return perf_stats_all

def BackTestPlot(account_value, 
                 baseline_start = config.START_TRADE_DATE, 
                 baseline_end = config.END_DATE, 
                 baseline_ticker = config.BASELINE_TICKER):

    df = account_value.copy()
    df = get_daily_return(df)

    bli, dow_strat = baseline_strat(ticker = baseline_ticker, 
                                    start = baseline_start, 
                                    end = baseline_end)      
    
    df['date'] = bli['date']
    
    DRL_strat = backtest_strat(df)
    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(returns = DRL_strat, benchmark_rets=dow_strat, set_context=False)

def backtest_strat(df):
    strategy_ret= df.copy()
    strategy_ret['date'] = pd.to_datetime(strategy_ret['date'])
    strategy_ret.set_index('date', drop = True, inplace = True)
    strategy_ret.index = strategy_ret.index.tz_localize('UTC')
    ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
    ts = ts.fillna(0.0)
    return ts.head(n=479)


def baseline_strat(ticker, start, end):
    bli = DataDowloader(start_date = start,
                     end_date = end,
                     ticker_list = [ticker]).fetch_data()
    bli['daily_return']=bli['close'].pct_change(1)
    dow_strat = backtest_strat(bli)
    return bli, dow_strat

def get_daily_return(df):
    df['daily_return']=df.account_value.pct_change(1)
    df=df.fillna(0.0)
    sharpe = ((252**0.5)*df['daily_return'].mean() / df['daily_return'].std()) if  df['daily_return'].std() != 0 else 0
    annual_return = ((df['daily_return'].mean()+1)**252-1)*100
    print("annual return: ", annual_return)
    print("sharpe ratio: ", sharpe)    
    return df