import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime
import os

from finrl.config import config
from finrl.marketdata.datadowloader import get_data_downloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split, data_filter
from finrl.env.environment import EnvSetup
from finrl.env.EnvMultipleStock_train import StockEnvTrain
from finrl.env.EnvMultipleStock_trade import StockEnvTrade
from finrl.env.EnvSingleStock import SingleStockEnv
from finrl.env.EnvPortfolio import StockPortfolioEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import BackTestStats, BackTestPlot, BaselineStats, BacktestResults


def backtest(model_name=config.SAVED_MODEL):
    """
    train an agent
    """
    matplotlib.use('WebAgg')
    
    DataDowloader = get_data_downloader(config.DATA_PROVIDER)

    if not os.path.exists(config.TRAINING_DATA_FILE):
        print("==============Start Fetching Data===========")
        df = DataDowloader(start_date = config.START_DATE,
                            end_date = config.END_DATE,
                            ticker_list = config.TICKER_LIST).fetch_data()
        print("==============Start Feature Engineering===========")
        df = FeatureEngineer(df,
                        use_technical_indicator=config.USE_TECHNICAL_INDICATOR,
                        user_defined_feature=config.USER_DEFINED_FEATURE,
                        use_turbulence=config.USE_TURBULENCE).preprocess_data()

        df.to_csv(config.TRAINING_DATA_FILE)
    else:
        print("==============Using Saved Data===========")
        df = pd.read_csv(config.TRAINING_DATA_FILE)
        selected_stocks = df.tic.unique()
        print('Selected tocks: {}'.format(', '.join(selected_stocks)))

    # Training & Trade data split
    train = data_split(df, config.START_DATE,config.START_TRADE_DATE)
    trade = data_split(df, config.START_TRADE_DATE,config.END_DATE)
    trade = data_filter(trade, config.MULTIPLE_STOCK_TICKER)

    # data normalization
    feaures_list = list(train.columns)
    #feaures_list.remove('date')
    #feaures_list.remove('tic')
    #feaures_list.remove('close')
    print('Features: {}'.format(', '.join(feaures_list)))
    #data_normaliser = preprocessing.StandardScaler()
    #train[feaures_list] = data_normaliser.fit_transform(train[feaures_list])
    #trade[feaures_list] = data_normaliser.fit_transform(trade[feaures_list])

    print("==============Enviroiment Setup===========")
    print("Trading policy: {}".format(config.TRADING_POLICY))
    train_env_class = StockEnvTrain
    trade_env_class = StockEnvTrade

    if config.TRADING_POLICY == 'SINGLE_STOCK':
        train_env_class = SingleStockEnv
        trade_env_class = SingleStockEnv
    
    if config.TRADING_POLICY == 'SINGLE_PORFOLIO':
        train_env_class = StockPortfolioEnv
        trade_env_class = StockPortfolioEnv

    # calculate state action space
    # stock_dimension = len(train.tic.unique())
    stock_dimension = config.NUMBER_SAMPLE_STOCKS
    state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension 

    print('Stock dimention: {}'.format(stock_dimension))
    print('State dimention {}'.format(state_space))

    env_setup = EnvSetup(stock_dim = stock_dimension,
                         population_space = config.NUMBER_OF_STOCKS,
                         sample_space = config.NUMBER_SAMPLE_STOCKS,
                         state_space = state_space,
                         hmax = config.MAXIMUM_STOCKS_PER_COMMIT,
                         hmin= config.STOCKS_PER_BATCH,
                         initial_amount = config.INITIAL_AMMOUNT,
                         transaction_cost_pct = config.TRANSACTION_COST_PCT)

    env_train = env_setup.create_env_training(data = train,
                                          env_class = train_env_class)
    agent = DRLAgent(env = env_train)

    print("==============Loading Model===========")
    print("Using agent {}".format(config.ENABLED_MODEL))
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    model_params_tuning=config.SAC_PARAMS
    model = agent.load_SAC(model_name = model_name, model_params = model_params_tuning)

    print("==============Start Trading===========")
    env_trade, obs_trade = env_setup.create_env_trading(data = trade,
                                         env_class = trade_env_class,
                                         turbulence_threshold=config.TURBULENCE_THRESHOLD) 

    df_account_value, df_actions = DRLAgent.DRL_prediction(model=model,
                                                          test_data = trade,
                                                          test_env = env_trade,
                                                          test_obs = obs_trade)

    print("============Simulate trading process==========")
    combined_data = pd.concat([trade, df_actions, df_account_value], axis=1, join="inner")
    BacktestResults(states=combined_data)


    print("==============Get Backtest Results===========")
    perf_stats_all = BackTestStats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')

    print("==============Get Baseline Stats===========")
    baesline_perf_stats = BaselineStats(baseline_ticker = config.BASELINE_TICKER, 
                                    baseline_start = config.START_TRADE_DATE,
                                    baseline_end = config.END_DATE)
    perf_stats_baesline = pd.DataFrame(baesline_perf_stats)
    perf_stats_baesline.to_csv("./"+config.RESULTS_DIR+"/perf_stats__baesline"+now+'.csv')

    print("==============Compare to {}===========".format(config.BASELINE_TICKER))
    BackTestPlot(df_account_value, 
                baseline_ticker = config.BASELINE_TICKER, 
                baseline_start = config.START_TRADE_DATE,
                baseline_end = config.END_DATE)

    if config.TRADING_POLICY == 'SINGLE_STOCK' or config.TRADING_POLICY == 'SINGLE_PORTFOLIO':
        print("==============Compare to AAPL itself buy-and-hold===========")
        BackTestPlot(account_value=df_account_value, baseline_ticker = config.BASELINE_TICKER)
    
    plt.plot()
    plt.show()

    df_account_value.to_csv("./"+config.RESULTS_DIR+"/df_account_value_"+now+'.csv')
    df_actions.to_csv("./"+config.RESULTS_DIR+"/df_actions_"+now+'.csv')