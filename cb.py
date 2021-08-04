import datetime
import json
import string
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import numpy as np
import pandas as pd
import requests

ENDPOINT = "https://api.pro.coinbase.com/products"
# Possible granularity values that api supports: ["60", "300", "900", "3600", "21600", "86400"],
# these are for ease of use during future programming
GLABELS = {"1m": "60", "5m": "300", "15m": "900", "1h": "3600", "6h": "21600", "24h": "86400"}
# Labels taken from https://docs.pro.coinbase.com/#get-historic-rates
CANDLABELS = ["time", "low", "high", "open", "close", "volume"]

def get_product(prod: str) -> json:
    ret = requests.get(ENDPOINT + f"/{prod}")
    if ret.status_code != 200:
        print(f"HTML Error: {ret.status_code}")
        raise ValueError
    return ret.json()


def get_all_product_information() -> [dict]:
    r = requests.get(ENDPOINT)
    if r.status_code != 200:
        print(f"HTML Error: {r.status_code}")
        raise ValueError
    return json.loads(r.content)


def get_product_ids() -> np.array([str]):
    data = get_all_product_information()
    ret = []
    for d in data:
        ret.append(d["id"])
    return np.array(ret)


def get_last_24hours(prod: str) -> json:
    ret = requests.get(ENDPOINT + f"/{prod}/stats")
    if ret.status_code != 200:
        print(f"HTML Error: {ret.status_code}")
        raise ValueError
    return ret.json()


def set_granularity(granularity: str = None) -> str:
    # See global declarations above for explanation of glabels
    g = granularity
    if granularity not in GLABELS.keys():
        g = "3600"
    else:
        g = GLABELS[granularity]
    return g


def get_num_candles(start: datetime.date, end: datetime.date = None, granularity: str = None) -> int:
    if end is None:
        end = datetime.datetime.now()
    td = (end - start).total_seconds()
    granularity = set_granularity(granularity)
    return int(td) // int(granularity)


def get_candles(prod: str, start: datetime.date, end: datetime.date = None, granularity: str = None) -> pd.DataFrame:
    # Granularity must be one of the following: ["1m", "5m", "15m", "1h", "6h", "24h"]
    # From API:  The maximum number of data points for a single request is 300 candles. If your selection of
    # start/end time and granularity will result in more than 300 data points, your request will be rejected. If you
    # wish to retrieve fine granularity data over a larger time range, you will need to make multiple requests with
    # new start/end ranges.
    if end is None:
        end = datetime.date.today()
    # See global declarations above for explanation of glabels
    granularity = set_granularity(granularity)
    num_candles = get_num_candles(start, end, granularity)
    df = pd.DataFrame(columns=CANDLABELS)

    while num_candles > 300:
        end = start + datetime.timedelta(seconds=(int(granularity)*300))
        params = {"start": start.isoformat(), "end": end.isoformat(), "granularity": granularity}
        r = requests.get(ENDPOINT + f"/{prod}/candles", params)
        if r.status_code != 200:
            print(f"HTML Error: {r.status_code}")
            raise ValueError
        temp = pd.DataFrame(r.json(), columns=CANDLABELS)
        # Results are returned from API with the most recent results first, i.e. end date is 0 entry, so reversing the
        # temp frame puts data in order
        df = pd.concat([df, temp.iloc[::-1]])
        start = end
        num_candles -= 300

    end = start + datetime.timedelta(seconds=(int(granularity)*num_candles))
    params = {"start": start.isoformat(), "end": end.isoformat(), "granularity": granularity}
    r = requests.get(ENDPOINT + f"/{prod}/candles", params)
    if r.status_code != 200:
        print(f"HTML Error: {r.status_code}")
        raise ValueError
    temp = pd.DataFrame(r.json(), columns=CANDLABELS)
    df = pd.concat([df, temp.iloc[::-1]])
    # Convert from sec since unix epoch (returned from API) to python datetime
    df["time"] = pd.to_datetime(df["time"], unit='s')
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: Dataframe to be augmented, should follow CANDLABELS
    :return: Augmented dataframe with log returns based on closing prices
    """
    # TODO: Implement parsing for returns based on different prices (open, high, low, etc.)
    return df.assign(log_return=lambda x: np.log(x.close / (x.close.shift(-1))))


def get_crypto_pair_with_returns(prod1: str, prod2: str,
                                 start: datetime.date, end: datetime.date = None,
                                 granularity: str = None) -> pd.DataFrame:
    df1 = compute_log_returns(get_candles(prod1, start, end, granularity))
    df2 = compute_log_returns(get_candles(prod2, start, end, granularity))
    return pd.merge(df1, df2, on="time", suffixes=(f"_{prod1}", f"_{prod2}"))


def plot_crypto_pair_returns(prod1: str, prod2: str,
                             start: datetime.date, end: datetime.date = None,
                             granularity: str = None):
    dfm = get_crypto_pair_with_returns(prod1, prod2, start, end, granularity)
    # https://stackoverflow.com/questions/23294197/plotting-chart-with-epoch-time-x-axis-using-matplotlib
    # Create figure and subplot axis
    fig, ax = plt.subplots()
    # Plot DataFrame to subplot axis (ax)
    dfm.plot(x="time", y=["log_return_BTC-USD", "log_return_ETH-USD"])
    # xtick format string
    date_fmt = '%d-%m-%y %H:%M:%S'
    # TODO: Look into set_major_formatter
    date_formatter = mdate.DateFormatter(date_fmt)
    ax.xaxis.set_major_formatter(date_formatter)
    # Automatically format dates, diagonally/daily
    fig.autofmt_xdate()
    plt.show()


def main():
    plot_crypto_pair_returns("BTC-USD", "ETH-USD", datetime.date(2021, 6, 1), granularity="1h")
    #print(get_candles("BTC-USD", datetime.date(2021, 7, 1), granularity="1h"))
