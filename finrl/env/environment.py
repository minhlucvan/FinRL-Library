from finrl.config import config
from stable_baselines3.common.vec_env import DummyVecEnv



class EnvSetup:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        stock_dim: int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount: int
            start money
        transaction_cost_pct : float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        tech_indicator_list: list
            a list of technical indicator names (modified from config.py)

    Methods
    -------
    create_env_training()
        create env class for training
    create_env_validation()
        create env class for validation
    create_env_trading()
        create env class for trading

    """
    def __init__(self, 
        stock_dim:int,
        state_space:int,
        hmax = 100,
        hmin = 100,
        sample_space = 3,
        population_space = 8,
        initial_amount = 1000000,
        transaction_cost_pct = 0.001,
        reward_scaling = 1e-4,
        result_dir='result/',
        tech_indicator_list = config.TECHNICAL_INDICATORS_LIST):

        self.result_dir = result_dir
        self.stock_dim = sample_space
        self.hmax = hmax
        self.hmin = hmin
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list
        # account balance + close price + shares + technical indicators
        self.state_space = state_space
        self.action_space = self.stock_dim
        self.sample_space = sample_space
        self.population_space = population_space


    def create_env_training(self, data, env_class, turbulence_threshold=100):
        env_train = DummyVecEnv([lambda: env_class(df = data,
                                                    stock_dim = self.stock_dim,
                                                    hmax = self.hmax,
                                                    hmin = self.hmin,
                                                    result_dir=self.result_dir,
                                                    sample_space = self.sample_space,
                                                    population_space = self.population_space,
                                                    initial_amount = self.initial_amount,
                                                    transaction_cost_pct = self.transaction_cost_pct,
                                                    reward_scaling = self.reward_scaling,
                                                    state_space = self.state_space,
                                                    action_space = self.action_space,
                                                    tech_indicator_list = self.tech_indicator_list,
                                                    turbulence_threshold = turbulence_threshold)])
        return env_train


    def create_env_validation(self, data, env_class, turbulence_threshold=100):
        env_validation = DummyVecEnv([lambda: env_class(df = data,
                                            stock_dim = self.stock_dim,
                                            hmax = self.hmax,
                                            hmin = self.hmin,
                                            result_dir=self.result_dir,
                                            sample_space = self.sample_space,
                                            population_space = self.population_space,
                                            initial_amount = self.initial_amount,
                                            transaction_cost_pct = self.transaction_cost_pct,
                                            reward_scaling = self.reward_scaling,
                                            state_space = self.state_space,
                                            action_space = self.action_space,
                                            tech_indicator_list = self.tech_indicator_list,
                                            turbulence_threshold = turbulence_threshold)])
        obs_validation = env_validation.reset()


        return env_validation, obs_validation

    def create_env_trading(self, env_class, data, turbulence_threshold=100):
        env_trade = DummyVecEnv([lambda: env_class(df = data,
                                            stock_dim = self.stock_dim,
                                            hmax = self.hmax,
                                            hmin = self.hmin,
                                            result_dir=self.result_dir,
                                            sample_space = self.sample_space,
                                            population_space = self.population_space,
                                            initial_amount = self.initial_amount,
                                            transaction_cost_pct = self.transaction_cost_pct,
                                            reward_scaling = self.reward_scaling,
                                            state_space = self.state_space,
                                            action_space = self.action_space,
                                            tech_indicator_list = self.tech_indicator_list,
                                            turbulence_threshold = turbulence_threshold)])
        obs_trade = env_trade.reset()


        return env_trade, obs_trade
