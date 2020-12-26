import pandas as pd
from .vnddowloader import VndDownloader
from .yahoodownloader import YahooDownloader

def get_data_downloader(provider):
    if(provider == 'vnd'):
        return VndDownloader

    return YahooDownloader


