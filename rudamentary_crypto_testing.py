import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import requests
import cb


class CryptoAsset:
    def __init__(self, _id: str = "BTC-USD",
                 _start: datetime.date = (datetime.date.today() - datetime.timedelta(days=30))):
        self.id: str = _id
        self.start: datetime.date = _start

    def get_candles(self, end: datetime.date = None, granularity: str = None) -> pd.DataFrame:
        # Defaults to hourly candles from one month prior
        return cb.compute_percent_returns(
               cb.compute_log_returns(
               cb.get_candles(self.id, self.start, end=end, granularity=granularity)))

    def get_summary(self, cols: str = None):
        return cb.compute_statistics(self.get_candles(), cols)


class CryptoPair:
    def __init__(self, _prod1: CryptoAsset, _prod2: CryptoAsset):
        assert (_prod1.start == _prod2.start)
        self.id1: str = getattr(_prod1, "id")
        self.id2: str = getattr(_prod2, "id")
        self.candles: pd.DataFrame = cb.get_crypto_pair_with_returns(self.id1, self.id2, _prod1.start)
        self.candles["log_ret_corr"] = cb.compute_moving_correlation(self.candles, [f"log_ret_{self.id1}",
                                                                                    f"log_ret_{self.id2}"])
        self.lr_close_corr: float = cb.compute_single_correlation(self.candles[f"close_{self.id1}"],
                                                                  self.candles[f"close_{self.id2}"])
        self.lr_log_ret_corr: float = cb.compute_single_correlation(self.candles[f"log_ret_{self.id1}"],
                                                                    self.candles[f"log_ret_{self.id2}"])
        self.lr_pct_ret_corr: float = cb.compute_single_correlation(self.candles[f"pct_ret_{self.id1}"],
                                                                    self.candles[f"pct_ret_{self.id2}"])



