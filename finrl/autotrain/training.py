import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
matplotlib.use('Agg')
import datetime

from finrl.config import config
from finrl.marketdata.datadowloader import get_data_downloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.environment import EnvSetup
from finrl.env.EnvMultipleStock_train import StockEnvTrain
from finrl.env.EnvMultipleStock_trade import StockEnvTrade
from finrl.model.models import DRLAgent
from finrl.trade.backtest import BackTestStats

DataDowloader = get_data_downloader(config.DATA_PROVIDER)

def train_one():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    df = DataDowloader(start_date = config.START_DATE,
                         end_date = config.END_DATE,
                         ticker_list = config.TICKER_LIST).fetch_data()
    print("==============Start Feature Engineering===========")
    df = FeatureEngineer(df,
                        use_technical_indicator=True,
                        use_turbulence=True).preprocess_data()

    df.to_csv(config.TRAINING_DATA_FILE)

    # Training & Trade data split
    train = data_split(df, config.START_DATE,config.START_TRADE_DATE)
    trade = data_split(df, config.START_TRADE_DATE,config.END_DATE)

    # data normalization
    feaures_list = list(train.columns)
    #feaures_list.remove('date')
    #feaures_list.remove('tic')
    #feaures_list.remove('close')
    print('features')
    print(', '.join(feaures_list))
    #data_normaliser = preprocessing.StandardScaler()
    #train[feaures_list] = data_normaliser.fit_transform(train[feaures_list])
    #trade[feaures_list] = data_normaliser.fit_transform(trade[feaures_list])

    # calculate state action space
    # stock_dimension = len(train.tic.unique())
    stock_dimension = config.NUMBER_SAMPLE_STOCKS
    state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension 

    env_setup = EnvSetup(stock_dim = stock_dimension,
                         population_space = config.NUMBER_OF_STOCKS,
                         sample_space = config.NUMBER_SAMPLE_STOCKS,
                         state_space = state_space,
                         hmax = config.MAXIMUM_STOCKS_PER_COMMIT,
                         hmin= config.STOCKS_PER_BATCH,
                         initial_amount = config.INITIAL_AMMOUNT,
                         transaction_cost_pct = config.TRANSACTION_COST_PCT)

    env_train = env_setup.create_env_training(data = train,
                                          env_class = StockEnvTrain)
    agent = DRLAgent(env = env_train)

    print("==============Model Training===========")
    print("Using Model {}".format(config.ENABLED_MODEL))
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    model_params_tuning=config.SAC_PARAMS
    model = agent.train_SAC(model_name = "SAC_{}".format(now), model_params = model_params_tuning)

    if config.ENABLED_MODEL == 'ppo':
        model_params_tuning=config.PPO_PARAMS
        model = agent.train_PPO(model_name = "PPO_{}".format(now), model_params = model_params_tuning)
    
    if config.ENABLED_MODEL == 'a2c':
        model_params_tuning=config.A2C_PARAMS
        model = agent.train_A2C(model_name = "A2C_{}".format(now), model_params = model_params_tuning)
    
    if config.ENABLED_MODEL == 'ddpg':
        model_params_tuning=config.DDPG_PARAMS
        model = agent.train_DDPG(model_name = "DDPG_{}".format(now), model_params = model_params_tuning)
    
    if config.ENABLED_MODEL == 'td3':
        model_params_tuning=config.TD3_PARAMS
        model = agent.train_TD3(model_name = "TD3_{}".format(now), model_params = model_params_tuning)

    print("==============Start Trading===========")
    env_trade, obs_trade = env_setup.create_env_trading(data = trade,
                                         env_class = StockEnvTrade,
                                         turbulence_threshold=250) 

    df_account_value,df_actions = DRLAgent.DRL_prediction(model=model,
                                                          test_data = trade,
                                                          test_env = env_trade,
                                                          test_obs = obs_trade)
    df_account_value.to_csv("./"+config.RESULTS_DIR+"/df_account_value_"+now+'.csv')
    df_actions.to_csv("./"+config.RESULTS_DIR+"/df_actions_"+now+'.csv')

    print("==============Get Backtest Results===========")
    perf_stats_all = BackTestStats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')