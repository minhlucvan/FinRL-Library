import pandas as pd
import requests
import os 
import datetime
import time
import finrl.config.config as config

def to_timestamp(date_str):
    timestanp = int(time.mktime(datetime.datetime.strptime(date_str, "%d/%m/%Y").timetuple()))
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

        stocks_df.to_csv(stocks_data_file, index=False, encoding='utf-8')
        print('saved stocks data to {}'.format(stocks_data_file))


    def get_stock_price_part(self, stock_code, start_date, end_date):
        start_time = to_timestamp(start_date)
        end_time = to_timestamp(end_date)
        params = {
            "resolution": 'D',
            "symbol": stock_code,
            "from": start_time,
            "to": end_time
        }
        url = 'https://dchart-api.vndirect.com.vn/dchart/history'

        print('retreving price history for {} period {} - {}'.format(stock_code, start_date, end_date))
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
        df = df.dropna()
        df = df.reset_index(drop=True)

        return df

    def get_stock_price_history(self, stock_code):
        periods = [
            ('01/01/2009', '31/12/2012'),
            ('01/01/2013', '31/12/2018'),
            ('01/01/2019', '01/12/2020')
        ]
        full_df = pd.DataFrame([])
        for period in periods:
            start_date, end_date = period
            period_df =  self.get_stock_price_part(stock_code, start_date, end_date)
            full_df = pd.concat([full_df, period_df])

        return full_df

    def load_trading_data(self, stocks_data_file, training_data_file):
        print('load stocks data from {}'.format(stocks_data_file))
        price_df = pd.DataFrame([])
        stocks_df = pd.read_csv(stocks_data_file)

        qualified_stocks_df = stocks_df.query('listedDate <= 20090101').query('status == "listed"').query('indexCode == "VN30"')

        print('number of qualified stocks {}'.format(qualified_stocks_df.shape))

        selected_stocks_tic_df = qualified_stocks_df.sample(n=self.stocks_dim)

        selected_stocks_tic = selected_stocks_tic_df['code'].tolist()

        selected_stocks_tic = config.TICKER_LIST

        for index, stock_code in  enumerate(selected_stocks_tic):
            print('{}/{} load stock data {}'.format(index, qualified_stocks_df.size,  stock_code))
            stock_df = self.get_stock_price_history(stock_code)
            price_df = pd.concat([ price_df, stock_df])

        print("Shape of DataFrame: ", price_df.shape)
        return price_df
