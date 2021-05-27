import numpy as np 
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class test(IStrategy):
    """
    The HLHB ("Huck loves her bucks!") System simply aims to catch short-term forex trends.
    More information in https://www.babypips.com/trading/forex-hlhb-system-explained
    """

    INTERFACE_VERSION = 2

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -10

    # Optimal timeframe for the strategy.
    timeframe = '4h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # EMA
        dataframe['ema12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema26'] = ta.EMA(dataframe, timeperiod=26)

        # SMA
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)

        # ERI
        dataframe['elder_ray_bull'] = dataframe['high'] -  dataframe['ema13']
        dataframe['elder_ray_bear'] = dataframe['low'] -  dataframe['ema13']
        # bear power’s value is negative but increasing (i.e. becoming less bearish)
        # bull power’s value is increasing (i.e. becoming more bullish)
        dataframe['eri_buy'] = ((dataframe['elder_ray_bear'] < 0) & (dataframe['elder_ray_bear'] > dataframe['elder_ray_bear'].shift(1))) | ((dataframe['elder_ray_bull'] > dataframe['elder_ray_bull'].shift(1))) 
        # bull power’s value is positive but decreasing (i.e. becoming less bullish)
        # bear power’s value is decreasing (i.e., becoming more bearish)
        dataframe['eri_sell'] = ((dataframe['elder_ray_bull'] > 0) & (dataframe['elder_ray_bear'] < dataframe['elder_ray_bear'].shift(1))) | ((dataframe['elder_ray_bull'] < dataframe['elder_ray_bull'].shift(1)))


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ema12'], dataframe['ema26'])) &
                (dataframe['sma50']> dataframe['sma200']) &
                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['eri_buy'] == True) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['ema12'], dataframe['ema26'])) &
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
    
