import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import requests

ENDPOINT = "https://api.pro.coinbase.com/products"
# Possible granularity values that api supports: ["60", "300", "900", "3600", "21600", "86400"],
# these are for ease of use during future programming
GRANULARITY_LABELS = {"1m": "60", "5m": "300", "15m": "900", "1h": "3600", "6h": "21600", "24h": "86400"}
# Labels taken from https://docs.pro.coinbase.com/#get-historic-rates
CANDLE_LABELS = ["time", "low", "high", "open", "close", "volume"]


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
    # See global declarations above for explanation of GRANULARITY_LABELS
    g = granularity
    if granularity not in GRANULARITY_LABELS.keys():
        g = "3600"
    else:
        g = GRANULARITY_LABELS[granularity]
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
    df = pd.DataFrame(columns=CANDLE_LABELS)

    while num_candles > 0:
        end = start + datetime.timedelta(seconds=(int(granularity) * 300))
        params = {"start": start.isoformat(), "end": end.isoformat(), "granularity": granularity}
        r = requests.get(ENDPOINT + f"/{prod}/candles", params)
        if r.status_code != 200:
            print(f"HTML Error: {r.status_code}")
            raise ValueError
        temp = pd.DataFrame(r.json(), columns=CANDLE_LABELS)
        # Results are returned from API with the most recent results first, i.e. end date is 0 entry, so reversing the
        # temp frame puts data in order
        df = pd.concat([df, temp.iloc[::-1]])
        start = end
        num_candles -= 300

    # end = start + datetime.timedelta(seconds=(int(granularity) * num_candles))
    # params = {"start": start.isoformat(), "end": end.isoformat(), "granularity": granularity}
    # r = requests.get(ENDPOINT + f"/{prod}/candles", params)
    # if r.status_code != 200:
    #     print(f"HTML Error: {r.status_code}")
    #     raise ValueError
    # temp = pd.DataFrame(r.json(), columns=CANDLE_LABELS)
    # df = pd.concat([df, temp.iloc[::-1]])
    # Convert from sec since unix epoch (returned from API) to python datetime
    df["time"] = pd.to_datetime(df["time"], unit='s')
    return df.set_index("time")


def compute_statistics(df: pd.DataFrame, cols: [str] = None, moment: int = 4) -> pd.DataFrame:
    """
    :param df: Pandas dataframe of data to have stats calculated
    :param cols: List of string names referring to columns of df to perform analysis on, defaults to conventional and
                 log return respectively
    :param moment: Number of moments to calculate, defaults to 4
    :return: Dataframe containing columns passed to cols of rows containing [min, max, std, (mean, variance, skew, kurtosis)]
    """
    if cols is None:
        cols = ["pct_ret", "log_ret"]
    params = dict()
    stats = ["min", "max", "std", "mean", "var", "skew", "kurt"]
    MAX_MOMENTS = 4
    if (moment > MAX_MOMENTS) or moment < 0:
        print(f"Invalid moment = {moment}, enter an int from 0 to 4")
        raise ValueError
    stats = stats[:len(stats) + (moment - MAX_MOMENTS)]
    for col in cols:
        params[col] = stats
    return df.agg(params)


def compute_correlation_matrix(u: pd.Series, v: pd.Series, method: str = "pearson") -> pd.DataFrame:
    """
    Intended to take in two Pandas Dataframe columns and return a 2x2 matrix of correlations between the columns
    :param u: First column, pandas.Series
    :param v: Second column, pandas.Series
    :param method: String in [‘pearson’, ‘kendall’, ‘spearman’], can also be upgraded to incorporate callable, for
    correlation between datasets only, not autocorrelation
    :return: 2x2 Matrix of correlations between the columns and themselves
    """
    uu = u.autocorr()
    uv = u.corr(v, method)
    vu = v.corr(u, method)
    vv = v.autocorr()
    df = pd.DataFrame([[uu, uv],
                       [vu, vv]],
                      index=[f"{u.name}", f"{v.name}"], columns=[f"{u.name}", f"{v.name}"])
    return df


def compute_single_correlation(u: pd.Series, v: pd.Series, method: str = "pearson") -> float:
    """
    Intended to take in two Pandas Dataframe columns and return a the correlation of the columns
    :param u: First column, pandas.Series
    :param v: Second column, pandas.Series
    :param method: String in [‘pearson’, ‘kendall’, ‘spearman’], can also be upgraded to incorporate callable, for
    correlation between datasets only, not autocorrelation
    :return: Float, the correlation between the two columns
    """
    return u.corr(v, method)


def compute_moving_correlation(df: pd.DataFrame, cols: [str], time_step: str = None) -> pd.Series:
    if time_step is None:
        time_step = "3D"
    if cols is None:
        print("Must specify columns to correlate")
        raise ValueError
    r = df[cols].rolling(time_step).corr()
    # TODO Assign column names to correlations dynamically
    r = r.rename(columns={cols[0]: f"{cols[0]}_corr", cols[1]: f"{cols[1]}_corr"})
    r = r.loc[(slice(None), f"{cols[0]}"), f"{cols[1]}_corr"]
    r = r.reset_index()
    r = r.drop(columns=["level_1"])
    r = r.set_index("time")
    return r


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: Dataframe to be augmented, should follow CANDLABELS
    :return: Augmented dataframe with log returns based on closing prices
    """
    # TODO: Implement parsing for returns based on different prices (open, high, low, etc.)
    df["log_ret"] = np.log(df.close) - np.log(df.close.shift(1))
    return df


def compute_percent_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: Dataframe to be augmented, should follow CANDLE_LABELS
    :return: Augmented dataframe with conventional returns based on closing prices
    """
    # TODO: Implement parsing for returns based on different prices (open, high, low, etc.)
    df["pct_ret"] = df.close.pct_change()
    return df


def get_crypto_pair_with_returns(prod1: str, prod2: str,
                                 start: datetime.date, end: datetime.date = None,
                                 granularity: str = None) -> pd.DataFrame:
    df1 = compute_percent_returns(compute_log_returns(get_candles(prod1, start, end, granularity)))
    df2 = compute_percent_returns(compute_log_returns(get_candles(prod2, start, end, granularity)))
    return pd.merge(df1, df2, on="time", suffixes=(f"_{prod1}", f"_{prod2}"))


def plot_crypto_pair_returns(prod1: str, prod2: str,
                             start: datetime.date, end: datetime.date = None,
                             granularity: str = None):
    granularity = set_granularity(granularity)
    dfm = get_crypto_pair_with_returns(prod1, prod2, start, end, granularity)
    # https://stackoverflow.com/questions/23294197/plotting-chart-with-epoch-time-x-axis-using-matplotlib
    # Create figure and subplot axis
    fig, ax = plt.subplots()
    # Plot DataFrame to subplot axis (ax)
    dfm.plot(y=[f"log_ret_{prod1}", f"log_ret_{prod2}"], use_index=True)
    # xtick format string
    date_fmt = '%d-%m-%y %H:%M:%S'
    # TODO: Look into set_major_formatter
    date_formatter = mdate.DateFormatter(date_fmt)
    ax.xaxis.set_major_formatter(date_formatter)
    # Automatically format dates, diagonally/daily
    fig.autofmt_xdate()
    plt.show()


def plot_crypto_pair_correlation(prod1: str, prod2: str,
                                 start: datetime.date, end: datetime.date = None,
                                 granularity: str = None, window: str = None):
    granularity = set_granularity(granularity)
    dfm = get_crypto_pair_with_returns(prod1, prod2, start, end, granularity)
    # TODO: Clean data properly from compute moving, multiindexing is a bitch
    # TODO: Funtionalize getting a pair with correlations as opposed to just having it here in plot
    dfm['log_ret_corr'] = compute_moving_correlation(dfm, [f"log_ret_{prod1}", f"log_ret_{prod2}"], window)[f"log_ret_{prod2}_corr"].values
    dfm['log_ret_corr_mean'] = np.full(dfm.index.shape, dfm.log_ret_corr.mean())
    # https://stackoverflow.com/questions/23294197/plotting-chart-with-epoch-time-x-axis-using-matplotlib
    # Create figure and subplot axis
    fig, ax = plt.subplots()
    # Plot DataFrame to subplot axis (ax)
    dfm.plot(y=["log_ret_corr", "log_ret_corr_mean"], use_index=True)
    # xtick format string
    date_fmt = '%d-%m-%y %H:%M:%S'
    # TODO: Look into set_major_formatter
    date_formatter = mdate.DateFormatter(date_fmt)
    ax.xaxis.set_major_formatter(date_formatter)
    # Automatically format dates, diagonally/daily
    fig.autofmt_xdate()
    plt.show()

# plot_crypto_pair_correlation("BTC-USD", "ETH-USD", datetime.date(2021, 6, 1))
def plot_return_histogram(prod1: str,
                          start: datetime.date, end: datetime.date = None,
                          granularity: str = None):
    # TODO: Implement visualization for statistical analysis
    pass


def main():
    df = get_candles("BTC-USD", datetime.date(2021, 6, 1), granularity="1h")
    print(compute_log_returns(df))
    plot_crypto_pair_returns("BTC-USD", "ETH-USD", datetime.date(2021, 6, 1))
    plot_crypto_pair_correlation("BTC-USD", "ETH-USD", datetime.date(2021, 6, 1))
    #df = compute_conventional_returns(df)
    #print(compute_statistics(df))
    #df = get_crypto_pair_with_returns("BTC-USD", "ETH-USD", datetime.date(2020, 6, 1), granularity="1h")
    #df = compute_moving_correlation(df, ["log_return_BTC-USD", "log_return_ETH-USD"])
    # https://pandas.pydata.org/docs/user_guide/advanced.html#advanced-xs
    #pl = df.loc[(slice(None), "log_return_ETH-USD"), :]
    #pl.reset_index().plot(x="time", y="log_return_BTC-USD_3D_corr")
    #plt.show()

