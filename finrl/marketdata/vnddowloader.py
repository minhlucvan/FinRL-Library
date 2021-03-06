import pandas as pd
import requests
import os 
import datetime
import time
from dateutil.relativedelta import relativedelta
import finrl.config.config as config

def to_timestamp(date_str):
    timestanp = int(time.mktime(datetime.datetime.strptime(date_str, "%Y-%m-%d").timetuple()))
    return timestanp

def to_date_str(timestamp):
    date = datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
    return date

def date_to_str(text):
    return text.replace("-", "")

class VndDownloader:

    def __init__(self, 
        start_date:str,
        end_date:str,
        ticker_list:list):

        self.stocks_data_file = config.MARKET_DATA_FILE
        self.training_data_file = config.TRAINING_DATA_FILE
        self.stocks_dim = config.NUMBER_OF_STOCKS

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list


    def fetch_data(self) -> pd.DataFrame:


        if not os.path.exists(self.stocks_data_file):
            self.load_stocks_data(self.stocks_data_file)

        data_df =  self.load_trading_data(self.stocks_data_file, self.training_data_file)

        return data_df

    def load_dataset(self, file_name: str) -> pd.DataFrame:
        _data = pd.read_csv(file_name)
        return _data

    def load_stocks_data(self, stocks_data_file):
        url = 'https://finfo-api.vndirect.com.vn/v4/stocks?q=type:STOCK~status:LISTED&fields=code,type,floor,isin,status,companyName,companyNameEng,shortName,listedDate,indexCode,industryName&size=3000'
        
        print('retriving data from {}'.format(url))
        response = requests.get(url=url)
        data = response.json() 
        
        stocks_data = data['data']
        print('got stocks data with {} elements'.format(len(stocks_data)))

        stocks_df = pd.DataFrame(stocks_data)
        
        stocks_df['listedDate'] = stocks_df['listedDate'].apply(date_to_str)

        stocks_df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)

        stocks_df.to_csv(stocks_data_file, index=False, encoding='utf-8')
        print('saved stocks data to {}'.format(stocks_data_file))


    def get_stock_price_part(self, stock_code, start_time, end_time):
        params = {
            "resolution": 'D',
            "symbol": stock_code,
            "from": start_time,
            "to": end_time
        }
        url = 'https://dchart-api.vndirect.com.vn/dchart/history'
        response = requests.get(url=url, params=params)
        data = response.json()

        columns = {
            "tic": stock_code,
            "date": data["t"],
            "close": data["c"],
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "volume": data["v"],
        }

        df = pd.DataFrame(columns)

        df['date'] = df['date'].astype(int)
        df['date'] = df['date'].apply(to_date_str)
        df['volume'] = df['volume'].astype(int)

        # drop missing data 
        df = df.fillna(0.0)
        df = df.reset_index(drop=True)

        return df

    def get_stock_price_history(self, stock_code):
        full_df = pd.DataFrame([])
        start_time = to_timestamp(self.start_date)
        end_time = to_timestamp(self.end_date)
        fetch_period = -1
        current_start_time = start_time
        current_end_time = start_time
        period_length = to_timestamp('2020-01-01') - to_timestamp('2019-01-01')

        while current_end_time < end_time:
            fetch_period += 1
            current_start_time = current_end_time
            current_end_time = min(current_start_time + period_length, end_time)
            period_df =  self.get_stock_price_part(stock_code, current_start_time, current_end_time)
            full_df = pd.concat([full_df, period_df], ignore_index=True)

        filterled_df = full_df.drop_duplicates()

        return filterled_df

    def load_trading_data(self, stocks_data_file, training_data_file):
        print('load stocks data from {}'.format(stocks_data_file))
        price_df = pd.DataFrame([])
        stocks_df = pd.read_csv(stocks_data_file)
        stocks_df['listedDate'] = pd.to_numeric(stocks_df['listedDate'])
        qualified_stocks_df = stocks_df.query('listedDate <= 20090101').query('status == "listed"')

        print('all stocks {}'.format(qualified_stocks_df['code'].count()))

        if not self.ticker_list:
            selected_stocks_tic_df = qualified_stocks_df.sample(n=self.stocks_dim)
            selected_stocks_tic = selected_stocks_tic_df['code'].tolist()
        else:
            selected_stocks_tic = self.ticker_list
        
    
        print('qualified stocks ({}) {}'.format(len(selected_stocks_tic), ', '.join(selected_stocks_tic)))
        
        for index, stock_code in  enumerate(selected_stocks_tic):
            print('{}/{} load stock data {}'.format(index, len(qualified_stocks_df.index),  stock_code))
            stock_df = self.get_stock_price_history(stock_code)
            price_df = pd.concat([ price_df, stock_df])

        filterled_df = price_df.drop_duplicates()
        print("Shape of DataFrame: ", filterled_df.shape)
        return filterled_df
