# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --------------------------------


class BBRSI(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/BbandRsi.cs

    """

    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    # ROI table:
    minimal_roi = {
        "0": 0.3,
        "88": 0.06,
        "145": 0.03,
        "289": 0.01
    }

    # Stoploss:
    stoploss = -0.3

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '1h'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    plot_config = {
        'main_plot': {
            'bb2_upperband': {'color': 'green'},
            'bb2_middleband': {'color': 'orange'},
            'bb2_lowerband': {'color': 'red'},
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'yellow'},
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger bands
        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb1_lowerband'] = bollinger1['lower']
        dataframe['bb1_middleband'] = bollinger1['mid']
        dataframe['bb1_upperband'] = bollinger1['upper']
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb2_lowerband'] = bollinger2['lower']
        dataframe['bb2_middleband'] = bollinger2['mid']
        dataframe['bb2_upperband'] = bollinger2['upper']
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb3_lowerband'] = bollinger3['lower']
        dataframe['bb3_middleband'] = bollinger3['mid']
        dataframe['bb3_upperband'] = bollinger3['upper']
        bollinger4 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['bb4_lowerband'] = bollinger4['lower']
        dataframe['bb4_middleband'] = bollinger4['mid']
        dataframe['bb4_upperband'] = bollinger4['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] < 40) &
                    (dataframe['close'] < dataframe['bb2_lowerband'])
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] > 80) &
                    (dataframe['close'] < dataframe['bb2_upperband'])
            ),
            'sell'] = 1
        return dataframe
