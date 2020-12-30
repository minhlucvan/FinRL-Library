import numpy as np
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from stockstats import StockDataFrame as Sdf
from finrl.config import config

pd.options.mode.chained_assignment = None

class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        df: DataFrame
            data downloaded from Yahoo API
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """
    def __init__(self, 
        df,
        use_technical_indicator=True,
        stocks_dim = config.NUMBER_OF_STOCKS,
        tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=False):

        self.df = df
        self.stocks_dim = stocks_dim
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_turbulence=use_turbulence
        self.user_defined_feature=user_defined_feature

        #type_list = self._get_type_list(5)
        #self.__features = type_list
        #self.__data_columns = config.DEFAULT_DATA_COLUMNS + self.__features


    def preprocess_data(self):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        df = self.df.copy()

        print("Droping missing values")
        df = self.drop_missing_values(df)

        # add technical indicators
        # stockstats require all 5 columns
        if (self.use_technical_indicator==True):
            # add technical indicators using stockstats
            print("Adding technical indicators")
            df=self.add_technical_indicator(df)

        # add turbulence index for multiple stock
        if self.use_turbulence==True:
            print("Adding turbulence index")
            df = self.add_turbulence(df)

        # add user defined feature
        if self.user_defined_feature == True:
            print("Adding user defined features")
            df = self.add_user_defined_feature(df)

       
        # fill the missing values at the beginning and the end
        print("Filling na values")
        df=df.fillna(method='bfill').fillna(method="ffill")
        return df

    def drop_missing_values(self, data):
        df = data.copy()
        dates = df.date.unique()
        df = df.set_index('date')
        data_df = pd.DataFrame([])

        for date in dates:
            date_df = ticker_df =  df.loc[date,:]
            if date_df.count() == self.stocks_dim:
                data_df = pd.concat([data_df, date_df])
        
        data_df = data_df.sort_values(['date','tic']).reset_index(drop=False)
        return data_df

    def add_technical_indicator(self, data):
        """
        calcualte technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.fillna(0.0)
        tickers = df.tic.unique()
        df = df.set_index('tic')
        full_indicators_df = pd.DataFrame([])


        for ticker in tickers:
            ticker_df =  df.loc[ticker,:]
            ticker_indicators_df = add_all_ta_features(ticker_df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
            full_indicators_df = pd.concat([full_indicators_df, ticker_indicators_df])

        full_indicators_df = full_indicators_df.sort_values(['date','tic']).reset_index(drop=False)

        return full_indicators_df

    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """          
        df = data.copy()
        df['daily_return']=df.close.pct_change(1)
        #df['return_lag_1']=df.close.pct_change(2)
        #df['return_lag_2']=df.close.pct_change(3)
        #df['return_lag_3']=df.close.pct_change(4)
        #df['return_lag_4']=df.close.pct_change(5)
        return df


    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calcualte_turbulence(df)
        df = df.merge(turbulence_index, on='date')
        df = df.sort_values(['date','tic']).reset_index(drop=True)
        return df


    def calcualte_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot=df.pivot(index='date', columns='tic', values='close')
        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0]*start
        #turbulence_index = [0]
        count=0
        for i in range(start,len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
            cov_temp = hist_price.cov()
            current_temp=(current_price - np.mean(hist_price,axis=0))
            temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
            if temp>0:
                count+=1
                if count>2:
                    turbulence_temp = temp[0][0]
                else:
                    #avoid large outlier because of the calculation just begins
                    turbulence_temp=0
            else:
                turbulence_temp=0
            turbulence_index.append(turbulence_temp)
        
        
        turbulence_index = pd.DataFrame({'date':df_price_pivot.index,
                                         'turbulence':turbulence_index})
        return turbulence_index

    def _get_type_list(self, feature_number):
        """
        :param feature_number: an int indicates the number of features
        :return: a list of features n
        """
        if feature_number == 1:
            type_list = ["close"]
        elif feature_number == 2:
            type_list = ["close", "volume"]
            #raise NotImplementedError("the feature volume is not supported currently")
        elif feature_number == 3:
            type_list = ["close", "high", "low"]
        elif feature_number == 4:
            type_list = ["close", "high", "low", "open"]
        elif feature_number == 5:
            type_list = ["close", "high", "low", "open","volume"]  
        else:
            raise ValueError("feature number could not be %s" % feature_number)
        return type_list