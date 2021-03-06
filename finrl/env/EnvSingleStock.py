import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SingleStockEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then 
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
        

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df,
                stock_dim,
                hmax,
                hmin,
                sample_space,
                population_space,     
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold,
                result_dir='result/',
                iteration=0,
                day = 0):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.population_space = population_space
        self.sample_space = sample_space
        self.df = df
        self.sample_tics = self.df['tic'].sample(n=self.sample_space).tolist()
        self.day = day
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.hmin = hmin
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.iteration = iteration

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_space,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:].set_index('tic').loc[self.sample_tics].reset_index() if self.sample_space > 1 else self.df.loc[self.day,:]
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold        
        # initalize state: inital amount + close price + shares + technical indicators + other features
        self.state = [self.initial_amount] + \
                      [self.data.close] + \
                      [0]*self.stock_dim  + \
                      sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])+ \
                      [self.data.open] + \
                      [self.data.high] + \
                      [self.data.low] +\
                      [self.data.daily_return] 
        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self.data.date]
        self.close_price_memory = [self.data.close]
        self.trades = 0
        self._seed()


    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        available_amount = self.state[2]
        action_amount = action * available_amount
        if  available_amount > 0:
            #update balance
            allowance_amount = min(abs(action_amount),self.state[index+self.stock_dim+1])
            transaction_amount = self.state[index+1]*allowance_amount
            self.state[0] += transaction_amount *  (1- self.transaction_cost_pct)

            self.state[index+self.stock_dim+1] -= allowance_amount
            self.cost +=self.state[index+1]*allowance_amount * self.transaction_cost_pct
            self.trades+=1
        else:
            pass

    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1]
        action_amount = action * available_amount
        # print('available_amount:{}'.format(available_amount))

        #update balance
        self.state[0] -= self.state[index+1]*min(available_amount, action_amount)*  (1+ self.transaction_cost_pct)

        self.state[index+self.stock_dim+1] += min(available_amount, action_amount)

        self.cost+=self.state[index+1]*min(available_amount, action_amount)* self.transaction_cost_pct
        self.trades+=1
        
    def step(self, actions):
        # print(self.day)
        # print(actions)
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig('results/account_value_train.png')
            plt.close()
            end_total_asset = self.state[0]+ self.state[1] * self.state[2]
            print("iteration result :{}".format(self.iteration))  
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.asset_memory)
            #df_total_value.to_csv('results/account_value_train.csv')
            print("total_reward: {}".format(self.state[0]+ self.state[1] * self.state[2]- self.initial_amount ))
            print("total_cost: ", self.cost)
            print("total_trades: ", self.trades)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            sharpe = ((252**0.5)*df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()) if df_total_value['daily_return'].std() != 0 else 0
            print("Sharpe: ", sharpe)
            print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            #df_rewards.to_csv('results/account_rewards_train.csv')
            
            
            return self.state, self.reward, self.terminal,{}

        else:
            # print(actions)
            actions = actions * self.hmax
            self.actions_memory.append(actions)
            #actions = (actions.astype(int))
            
            begin_total_asset = self.state[0] + self.state[1]*self.state[2]
            # print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            if actions[0] < 0:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(0, actions[0])

            if actions[0] > 0:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(0, actions[0])

            self.day += 1
            self.data = self.df.loc[self.day,:].set_index('tic').loc[self.sample_tics].reset_index() if self.sample_space > 1 else self.df.loc[self.day,:]
            #load next state
            # print("stock_shares:{}".format(self.state))
            self.state =  [self.state[0]] + \
                    [self.data.close] + \
                    [self.state[2]] + \
                      sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])+ \
                      [self.data.open] + \
                      [self.data.high] + \
                      [self.data.low] +\
                      [self.data.daily_return] 
            
            end_total_asset = self.state[0]+ self.state[1] *self.state[2]
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self.data.date)
            self.close_price_memory.append(self.data.close)

            # print("end_total_asset:{}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward*self.reward_scaling



        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.sample_tics = self.df['tic'].sample(n=self.sample_space).tolist()
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:].set_index('tic').loc[self.sample_tics].reset_index() if self.sample_space > 1 else self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self.data.date]
        self.iteration += 1 
        #initiate state
        self.state = [self.initial_amount] + \
                      [self.data.close] + \
                      [0]*self.stock_dim + \
                      sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])+ \
                      [self.data.open] + \
                      [self.data.high] + \
                      [self.data.low] +\
                      [self.data.daily_return] 
        return self.state
    
    def render(self, mode='human'):
        return self.state
    
    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory[:-1]
        close_price_list = self.close_price_memory[:-1]

        action_list = self.actions_memory
        df_actions = pd.DataFrame({'date':date_list,'actions':action_list,'close_price':close_price_list})
        df_actions['daily_return']=df_actions.close_price.pct_change()
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]