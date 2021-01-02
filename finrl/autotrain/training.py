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
from finrl.model.models import DRLAgent
from finrl.autotrain.backtesting import backtest
from finrl.env.EnvSingleStock import SingleStockEnv
from finrl.env.EnvPortfolio import StockPortfolioEnv

DataDowloader = get_data_downloader(config.DATA_PROVIDER)

def train_one():
    """
    train an agent
    """

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
    print('features', ', '.join(feaures_list))
    #data_normaliser = preprocessing.StandardScaler()
    #train[feaures_list] = data_normaliser.fit_transform(train[feaures_list])
    #trade[feaures_list] = data_normaliser.fit_transform(trade[feaures_list])

    print("==============Enviroiment Setup===========")
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
    stock_data_dimension = len(config.STOCK_DATA_COLUMNS)
    tech_indicators_dimension = len(config.TECHNICAL_INDICATORS_LIST)
    user_defined_dimension = len(config.STOCK_USER_DEFINED_COLUMNS)
    state_space = 1 + (1 + user_defined_dimension + tech_indicators_dimension + stock_data_dimension)*stock_dimension

    print('Stock dimension: {}'.format(stock_dimension))
    print('State dimension {}'.format(state_space))
    
    env_setup = EnvSetup(stock_dim = stock_dimension,
                         population_space = config.NUMBER_OF_STOCKS,
                         sample_space = config.NUMBER_SAMPLE_STOCKS,
                         state_space = state_space,
                         hmax = config.MAXIMUM_STOCKS_PER_COMMIT,
                         hmin= config.STOCKS_PER_BATCH,
                         initial_amount = config.INITIAL_AMMOUNT,
                         transaction_cost_pct = config.TRANSACTION_COST_PCT)

    env_train = env_setup.create_env_training(data = train,
                                          env_class = train_env_class,
                                          turbulence_threshold=config.TURBULENCE_THRESHOLD)
    agent = DRLAgent(env = env_train)


    
    print("==============Model Training===========")
    print("Using Model {}".format(config.ENABLED_MODEL))
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    model_params_tuning=config.SAC_PARAMS
    model_name = "SAC_{}".format(now)
    model = agent.train_SAC(model_name = model_name, model_params = model_params_tuning)

    if config.ENABLED_MODEL == 'ppo':
        model_params_tuning=config.PPO_PARAMS
        model_name = "PPO_{}".format(now)
        model = agent.train_PPO(model_name=model_name, model_params = model_params_tuning)
    
    if config.ENABLED_MODEL == 'a2c':
        model_params_tuning=config.A2C_PARAMS
        model_name = "A2C_{}".format(now)
        model = agent.train_A2C(model_name=model_name, model_params = model_params_tuning)
    
    if config.ENABLED_MODEL == 'ddpg':
        model_params_tuning=config.DDPG_PARAMS
        model_name = "DDPG_{}".format(now)
        model = agent.train_DDPG(model_name=model_name, model_params = model_params_tuning)
    
    if config.ENABLED_MODEL == 'td3':
        model_params_tuning=config.TD3_PARAMS
        model_name = "TD3_{}".format(now)
        model = agent.train_TD3(model_name=model_name, model_params = model_params_tuning)

    print("==============Model Testing===========")
    backtest(model_name=model_name)
    
