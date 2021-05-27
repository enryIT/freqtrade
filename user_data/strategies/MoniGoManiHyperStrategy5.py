# --- Do not remove these libs ----------------------------------------------------------------------
from scipy.interpolate import interp1d

import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging
import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from datetime import datetime, timedelta

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy \
    import IStrategy, CategoricalParameter, IntParameter, merge_informative_pair, timeframe_to_minutes
from freqtrade.state import RunMode
from numpy import timedelta64
from pandas import DataFrame

logger = logging.getLogger(__name__)


# ^ TA-Lib Autofill mostly broken in JetBrains Products,
# ta._ta_lib.<function_name> can temporarily be used while writing as a workaround
# Then change back to ta.<function_name> so IDE won't nag about accessing a protected member of TA-Lib
# ----------------------------------------------------------------------------------------------------


class MoniGoManiHyperStrategy5(IStrategy):
    """
    ####################################################################################
    ####                                                                            ####
    ###                         MoniGoMani v0.10.0 by Rikj000                        ###
    ##                          -----------------------------                         ##
    #               Isn't that what we all want? Our money to go many?                 #
    #          Well that's what this Freqtrade strategy hopes to do for you!           #
    ##       By giving you/HyperOpt a lot of signals to alter the weight from         ##
    ###           ------------------------------------------------------             ###
    ##        Big thank you to xmatthias and everyone who helped on MoniGoMani,       ##
    ##      Freqtrade Discord support was also really helpful so thank you too!       ##
    ###         -------------------------------------------------------              ###
    ##              Disclaimer: This strategy is under development.                   ##
    #      I do not recommend running it live until further development/testing.       #
    ##                      TEST IT BEFORE USING IT!                                  ##
    ###                                                              ▄▄█▀▀▀▀▀█▄▄     ###
    ##               -------------------------------------         ▄█▀  ▄ ▄    ▀█▄    ##
    ###   If you like my work, feel free to donate or use one of   █   ▀█▀▀▀▀▄   █   ###
    ##   my referral links, that would also greatly be appreciated █    █▄▄▄▄▀   █    ##
    #     ICONOMI: https://www.iconomi.com/register?ref=JdFzz      █    █    █   █     #
    ##  Binance: https://www.binance.com/en/register?ref=97611461  ▀█▄ ▀▀█▀█▀  ▄█▀    ##
    ###          BTC: 19LL2LCMZo4bHJgy15q1Z1bfe7mV4bfoWK             ▀▀█▄▄▄▄▄█▀▀     ###
    ####                                                                            ####
    ####################################################################################
    """

    # If enabled all Weighted Signal results will be added to the dataframe for easy debugging with BreakPoints
    # Warning: Disable this for anything else then debugging in an IDE! (Integrated Development Environment)
    debuggable_weighted_signal_dataframe = False

    # If enabled MoniGoMani logging will be displayed to the console and be integrated in Freqtrades native logging
    # For live it's recommended to disable at least info/debug logging, to keep MGM as lightweight as possible!
    use_mgm_logging = True
    mgm_log_levels_enabled = {
        'info': True,
        'warning': True,
        'error': True,
        'debug': False
        # ^ Debug is very verbose! Always set it to False when BackTesting/HyperOpting!
        # (Only recommended to be True in an IDE with Breakpoints enabled or when you suspect a bug in the code)
    }

    # Ps: Documentation has been moved to the Buy/Sell HyperOpt Space Parameters sections below this copy-paste section
    ####################################################################################################################
    #                                    START OF HYPEROPT RESULTS COPY-PASTE SECTION                                  #
    ####################################################################################################################

    # Buy hyperspace params:
    buy_params = {
        "buy__downwards_trend_total_signal_needed": 175,
        "buy__downwards_trend_total_signal_needed_candles_lookback_window": 4,
        "buy__sideways_trend_total_signal_needed": 478,
        "buy__sideways_trend_total_signal_needed_candles_lookback_window": 2,
        "buy__upwards_trend_total_signal_needed": 50,
        "buy__upwards_trend_total_signal_needed_candles_lookback_window": 6,
        "buy_downwards_trend_adx_strong_up_weight": 99,
        "buy_downwards_trend_bollinger_bands_weight": 38,
        "buy_downwards_trend_ema_long_golden_cross_weight": 92,
        "buy_downwards_trend_ema_short_golden_cross_weight": 25,
        "buy_downwards_trend_macd_weight": 73,
        "buy_downwards_trend_rsi_weight": 51,
        "buy_downwards_trend_sma_long_golden_cross_weight": 49,
        "buy_downwards_trend_sma_short_golden_cross_weight": 91,
        "buy_downwards_trend_vwap_cross_weight": 29,
        "buy_sideways_trend_adx_strong_up_weight": 76,
        "buy_sideways_trend_bollinger_bands_weight": 58,
        "buy_sideways_trend_ema_long_golden_cross_weight": 54,
        "buy_sideways_trend_macd_weight": 74,
        "buy_sideways_trend_sma_long_golden_cross_weight": 37,
        "buy_sideways_trend_vwap_cross_weight": 79,
        "buy_upwards_trend_adx_strong_up_weight": 75,
        "buy_upwards_trend_bollinger_bands_weight": 96,
        "buy_upwards_trend_ema_long_golden_cross_weight": 42,
        "buy_upwards_trend_ema_short_golden_cross_weight": 21,
        "buy_upwards_trend_macd_weight": 68,
        "buy_upwards_trend_rsi_weight": 86,
        "buy_upwards_trend_sma_long_golden_cross_weight": 52,
        "buy_upwards_trend_sma_short_golden_cross_weight": 17,
        "buy_upwards_trend_vwap_cross_weight": 9,
        "buy___trades_when_downwards": True,  # value loaded from strategy
        "buy___trades_when_sideways": False,  # value loaded from strategy
        "buy___trades_when_upwards": True,  # value loaded from strategy
        "buy_sideways_trend_ema_short_golden_cross_weight": 100,  # value loaded from strategy
        "buy_sideways_trend_rsi_weight": 0,  # value loaded from strategy
        "buy_sideways_trend_sma_short_golden_cross_weight": 100,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "sell___unclogger_minimal_losing_trade_duration_minutes": 30,
        "sell___unclogger_minimal_losing_trades_open": 5,
        "sell___unclogger_open_trades_losing_percentage_needed": 50,
        "sell___unclogger_trend_lookback_candles_window": 53,
        "sell___unclogger_trend_lookback_candles_window_percentage_needed": 22,
        "sell__downwards_trend_total_signal_needed": 638,
        "sell__downwards_trend_total_signal_needed_candles_lookback_window": 4,
        "sell__sideways_trend_total_signal_needed": 779,
        "sell__sideways_trend_total_signal_needed_candles_lookback_window": 4,
        "sell__upwards_trend_total_signal_needed": 513,
        "sell__upwards_trend_total_signal_needed_candles_lookback_window": 2,
        "sell_downwards_trend_adx_strong_down_weight": 42,
        "sell_downwards_trend_bollinger_bands_weight": 98,
        "sell_downwards_trend_ema_long_death_cross_weight": 2,
        "sell_downwards_trend_ema_short_death_cross_weight": 89,
        "sell_downwards_trend_macd_weight": 39,
        "sell_downwards_trend_rsi_weight": 25,
        "sell_downwards_trend_sma_long_death_cross_weight": 62,
        "sell_downwards_trend_sma_short_death_cross_weight": 72,
        "sell_downwards_trend_vwap_cross_weight": 86,
        "sell_sideways_trend_adx_strong_down_weight": 42,
        "sell_sideways_trend_bollinger_bands_weight": 61,
        "sell_sideways_trend_ema_long_death_cross_weight": 65,
        "sell_sideways_trend_ema_short_death_cross_weight": 75,
        "sell_sideways_trend_macd_weight": 36,
        "sell_sideways_trend_rsi_weight": 70,
        "sell_sideways_trend_sma_long_death_cross_weight": 10,
        "sell_sideways_trend_sma_short_death_cross_weight": 84,
        "sell_sideways_trend_vwap_cross_weight": 0,
        "sell_upwards_trend_adx_strong_down_weight": 50,
        "sell_upwards_trend_bollinger_bands_weight": 91,
        "sell_upwards_trend_ema_long_death_cross_weight": 26,
        "sell_upwards_trend_ema_short_death_cross_weight": 15,
        "sell_upwards_trend_macd_weight": 14,
        "sell_upwards_trend_rsi_weight": 1,
        "sell_upwards_trend_sma_long_death_cross_weight": 73,
        "sell_upwards_trend_sma_short_death_cross_weight": 12,
        "sell_upwards_trend_vwap_cross_weight": 53,
        "sell___trades_when_downwards": True,  # value loaded from strategy
        "sell___trades_when_sideways": False,  # value loaded from strategy
        "sell___trades_when_upwards": True,  # value loaded from strategy
        "sell___unclogger_enabled": True,  # value loaded from strategy
        "sell___unclogger_trend_lookback_window_uses_downwards_candles": True,  # value loaded from strategy
        "sell___unclogger_trend_lookback_window_uses_sideways_candles": True,  # value loaded from strategy
        "sell___unclogger_trend_lookback_window_uses_upwards_candles": False,  # value loaded from strategy
    }

    # ROI table:
    minimal_roi = {
        "0": 0.461,
        "5": 0.45684,
        "10": 0.45268,
        "15": 0.44852,
        "20": 0.44436,
        "25": 0.4402,
        "30": 0.43604,
        "35": 0.43188,
        "40": 0.42773,
        "45": 0.42357,
        "50": 0.41941,
        "55": 0.41525,
        "60": 0.41109,
        "65": 0.40693,
        "70": 0.40277,
        "75": 0.39861,
        "80": 0.39445,
        "85": 0.39029,
        "90": 0.38613,
        "95": 0.38197,
        "100": 0.37781,
        "105": 0.37365,
        "110": 0.3695,
        "115": 0.36534,
        "120": 0.36118,
        "125": 0.35702,
        "130": 0.35286,
        "135": 0.3487,
        "140": 0.34454,
        "145": 0.34038,
        "150": 0.33622,
        "155": 0.33206,
        "160": 0.3279,
        "165": 0.32374,
        "170": 0.31958,
        "175": 0.31542,
        "180": 0.31127,
        "185": 0.30711,
        "190": 0.30295,
        "195": 0.29879,
        "200": 0.29463,
        "205": 0.29047,
        "210": 0.28631,
        "215": 0.28215,
        "220": 0.27799,
        "225": 0.27383,
        "230": 0.26967,
        "235": 0.26551,
        "240": 0.26135,
        "245": 0.25719,
        "250": 0.25304,
        "255": 0.24888,
        "260": 0.24472,
        "265": 0.24056,
        "270": 0.2364,
        "275": 0.23224,
        "280": 0.22808,
        "285": 0.22392,
        "290": 0.21976,
        "295": 0.2156,
        "300": 0.21144,
        "305": 0.20728,
        "310": 0.20312,
        "315": 0.19896,
        "320": 0.19481,
        "325": 0.19065,
        "330": 0.18649,
        "335": 0.18233,
        "340": 0.17877,
        "345": 0.1776,
        "350": 0.17643,
        "355": 0.17526,
        "360": 0.1741,
        "365": 0.17293,
        "370": 0.17176,
        "375": 0.17059,
        "380": 0.16942,
        "385": 0.16826,
        "390": 0.16709,
        "395": 0.16592,
        "400": 0.16475,
        "405": 0.16358,
        "410": 0.16242,
        "415": 0.16125,
        "420": 0.16008,
        "425": 0.15891,
        "430": 0.15775,
        "435": 0.15658,
        "440": 0.15541,
        "445": 0.15424,
        "450": 0.15307,
        "455": 0.15191,
        "460": 0.15074,
        "465": 0.14957,
        "470": 0.1484,
        "475": 0.14724,
        "480": 0.14607,
        "485": 0.1449,
        "490": 0.14373,
        "495": 0.14256,
        "500": 0.1414,
        "505": 0.14023,
        "510": 0.13906,
        "515": 0.13789,
        "520": 0.13672,
        "525": 0.13556,
        "530": 0.13439,
        "535": 0.13322,
        "540": 0.13205,
        "545": 0.13089,
        "550": 0.12972,
        "555": 0.12855,
        "560": 0.12738,
        "565": 0.12621,
        "570": 0.12505,
        "575": 0.12388,
        "580": 0.12271,
        "585": 0.12154,
        "590": 0.12038,
        "595": 0.11921,
        "600": 0.11804,
        "605": 0.11687,
        "610": 0.1157,
        "615": 0.11454,
        "620": 0.11337,
        "625": 0.1122,
        "630": 0.11103,
        "635": 0.10987,
        "640": 0.1087,
        "645": 0.10753,
        "650": 0.10636,
        "655": 0.10519,
        "660": 0.10403,
        "665": 0.10286,
        "670": 0.10169,
        "675": 0.10052,
        "680": 0.09935,
        "685": 0.09819,
        "690": 0.09702,
        "695": 0.09585,
        "700": 0.09468,
        "705": 0.09352,
        "710": 0.09235,
        "715": 0.09118,
        "720": 0.09001,
        "725": 0.08884,
        "730": 0.08768,
        "735": 0.08651,
        "740": 0.08534,
        "745": 0.08417,
        "750": 0.08301,
        "755": 0.08184,
        "760": 0.08067,
        "765": 0.0795,
        "770": 0.07833,
        "775": 0.07717,
        "780": 0.076,
        "785": 0.07483,
        "790": 0.07366,
        "795": 0.07249,
        "800": 0.07133,
        "805": 0.07016,
        "810": 0.06899,
        "815": 0.06782,
        "820": 0.06666,
        "825": 0.06549,
        "830": 0.06432,
        "835": 0.06315,
        "840": 0.06198,
        "845": 0.06082,
        "850": 0.05965,
        "855": 0.05848,
        "860": 0.05731,
        "865": 0.05615,
        "870": 0.05498,
        "875": 0.05381,
        "880": 0.05264,
        "885": 0.05147,
        "890": 0.05031,
        "895": 0.04914,
        "900": 0.04797,
        "905": 0.0468,
        "910": 0.04563,
        "915": 0.04447,
        "920": 0.04385,
        "925": 0.04361,
        "930": 0.04336,
        "935": 0.04312,
        "940": 0.04288,
        "945": 0.04263,
        "950": 0.04239,
        "955": 0.04214,
        "960": 0.0419,
        "965": 0.04165,
        "970": 0.04141,
        "975": 0.04116,
        "980": 0.04092,
        "985": 0.04068,
        "990": 0.04043,
        "995": 0.04019,
        "1000": 0.03994,
        "1005": 0.0397,
        "1010": 0.03945,
        "1015": 0.03921,
        "1020": 0.03896,
        "1025": 0.03872,
        "1030": 0.03848,
        "1035": 0.03823,
        "1040": 0.03799,
        "1045": 0.03774,
        "1050": 0.0375,
        "1055": 0.03725,
        "1060": 0.03701,
        "1065": 0.03676,
        "1070": 0.03652,
        "1075": 0.03628,
        "1080": 0.03603,
        "1085": 0.03579,
        "1090": 0.03554,
        "1095": 0.0353,
        "1100": 0.03505,
        "1105": 0.03481,
        "1110": 0.03456,
        "1115": 0.03432,
        "1120": 0.03408,
        "1125": 0.03383,
        "1130": 0.03359,
        "1135": 0.03334,
        "1140": 0.0331,
        "1145": 0.03285,
        "1150": 0.03261,
        "1155": 0.03236,
        "1160": 0.03212,
        "1165": 0.03188,
        "1170": 0.03163,
        "1175": 0.03139,
        "1180": 0.03114,
        "1185": 0.0309,
        "1190": 0.03065,
        "1195": 0.03041,
        "1200": 0.03016,
        "1205": 0.02992,
        "1210": 0.02968,
        "1215": 0.02943,
        "1220": 0.02919,
        "1225": 0.02894,
        "1230": 0.0287,
        "1235": 0.02845,
        "1240": 0.02821,
        "1245": 0.02796,
        "1250": 0.02772,
        "1255": 0.02748,
        "1260": 0.02723,
        "1265": 0.02699,
        "1270": 0.02674,
        "1275": 0.0265,
        "1280": 0.02625,
        "1285": 0.02601,
        "1290": 0.02576,
        "1295": 0.02552,
        "1300": 0.02528,
        "1305": 0.02503,
        "1310": 0.02479,
        "1315": 0.02454,
        "1320": 0.0243,
        "1325": 0.02405,
        "1330": 0.02381,
        "1335": 0.02356,
        "1340": 0.02332,
        "1345": 0.02308,
        "1350": 0.02283,
        "1355": 0.02259,
        "1360": 0.02234,
        "1365": 0.0221,
        "1370": 0.02185,
        "1375": 0.02161,
        "1380": 0.02136,
        "1385": 0.02112,
        "1390": 0.02088,
        "1395": 0.02063,
        "1400": 0.02039,
        "1405": 0.02014,
        "1410": 0.0199,
        "1415": 0.01965,
        "1420": 0.01941,
        "1425": 0.01916,
        "1430": 0.01892,
        "1435": 0.01868,
        "1440": 0.01843,
        "1445": 0.01819,
        "1450": 0.01794,
        "1455": 0.0177,
        "1460": 0.01745,
        "1465": 0.01721,
        "1470": 0.01696,
        "1475": 0.01672,
        "1480": 0.01648,
        "1485": 0.01623,
        "1490": 0.01599,
        "1495": 0.01574,
        "1500": 0.0155,
        "1505": 0.01525,
        "1510": 0.01501,
        "1515": 0.01476,
        "1520": 0.01452,
        "1525": 0.01428,
        "1530": 0.01403,
        "1535": 0.01379,
        "1540": 0.01354,
        "1545": 0.0133,
        "1550": 0.01305,
        "1555": 0.01281,
        "1560": 0.01256,
        "1565": 0.01232,
        "1570": 0.01208,
        "1575": 0.01183,
        "1580": 0.01159,
        "1585": 0.01134,
        "1590": 0.0111,
        "1595": 0.01085,
        "1600": 0.01061,
        "1605": 0.01036,
        "1610": 0.01012,
        "1615": 0.00988,
        "1620": 0.00963,
        "1625": 0.00939,
        "1630": 0.00914,
        "1635": 0.0089,
        "1640": 0.00865,
        "1645": 0.00841,
        "1650": 0.00816,
        "1655": 0.00792,
        "1660": 0.00768,
        "1665": 0.00743,
        "1670": 0.00719,
        "1675": 0.00694,
        "1680": 0.0067,
        "1685": 0.00645,
        "1690": 0.00621,
        "1695": 0.00596,
        "1700": 0.00572,
        "1705": 0.00548,
        "1710": 0.00523,
        "1715": 0.00499,
        "1720": 0.00474,
        "1725": 0.0045,
        "1730": 0.00425,
        "1735": 0.00401,
        "1740": 0.00376,
        "1745": 0.00352,
        "1750": 0.00328,
        "1755": 0.00303,
        "1760": 0.00279,
        "1765": 0.00254,
        "1770": 0.0023,
        "1775": 0.00205,
        "1780": 0.00181,
        "1785": 0.00156,
        "1790": 0.00132,
        "1795": 0.00108,
        "1800": 0.00083,
        "1805": 0.00059,
        "1810": 0.00034,
        "1815": 0.0001,
        "1820": 0
    }

    # Stoploss:
    stoploss = -0.218

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.013
    trailing_only_offset_is_reached = True

    ####################################################################################################################
    #                                     END OF HYPEROPT RESULTS COPY-PASTE SECTION                                   #
    ####################################################################################################################

    # Create dictionary to store custom information MoniGoMani will be using in RAM
    custom_info = {
        'open_trades': {}
    }

    # If enabled MoniGoMani's custom stoploss function will be used (Needed for open_trade custom_information_storage)
    use_custom_stoploss = True  # Leave this enabled when using the 'losing trade unclogger'

    # Create class level runmode detection (No need for configuration, will automatically be detected,
    # changed & used at runtime)
    is_dry_live_run_detected = True

    # TimeFrame-Zoom:
    # To prevent profit exploitation during backtesting/hyperopting we backtest/hyperopt this can be used.
    # When normally a 'timeframe' (1h candles) would be used, you can zoom in using a smaller 'backtest_timeframe'
    # (5m candles) instead. This happens while still using an 'informative_timeframe' (original 1h candles) to generate
    # the buy/sell signals.

    # With this more realistic results should be found during backtesting/hyperopting. Since the buy/sell signals will 
    # operate on the same 'timeframe' that live would use (1h candles), while at the same time 'backtest_timeframe' 
    # (5m or 1m candles) will simulate price movement during that 'timeframe' (1h candle), providing more realistic 
    # trailing stoploss and ROI behaviour during backtesting/hyperopting.

    # Warning: Since MoniGoMani v0.10.0 it appears TimeFrame-Zoom is not needed anymore and even lead to bad results!
    # Warning: Candle data for both 'timeframe' as 'backtest_timeframe' will have to be downloaded before you will be
    # able to backtest/hyperopt! (Since both will be used)
    # Warning: This will be slower than backtesting at 1h and 1m is a CPU killer. But if you plan on using trailing
    # stoploss or ROI, you probably want to know that your backtest results are not complete lies.
    # Source: https://brookmiles.github.io/freqtrade-stuff/2021/04/12/backtesting-traps/

    # To disable TimeFrame-Zoom just use the same candles for 'timeframe' & 'backtest_timeframe'
    timeframe = '1h'  # Optimal TimeFrame for MoniGoMani (used during Dry/Live-Runs)
    backtest_timeframe = '1h'  # Optimal TimeFrame-Zoom for MoniGoMani (used to zoom in during Backtesting/HyperOpting)
    informative_timeframe = timeframe

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals.
    # In live and dry runs this ratio will be 1, so nothing changes there.
    # But we need `startup_candle_count` to be for the timeframe of 
    # `informative_timeframe` (1h) not `timeframe` (5m) for backtesting.
    startup_candle_count: int = 400 * int(timeframe_to_minutes(informative_timeframe) / timeframe_to_minutes(timeframe))
    # SMA200 needs 200 candles before producing valid signals
    # EMA200 needs an extra 200 candles of SMA200 before producing valid signals

    # Precision:
    # This value can be used to control the precision of hyperopting.
    # A value of 1/5 will effectively set the step size to be 5 (0, 5, 10 ...)
    # A value of 5 will set the step size to be 1/5=0.2 (0, 0.2, 0.4, 0.8, ...)
    # A smaller value will limit the search space a lot, but may skip over good values.
    precision = 1

    # Number of weighted signals:
    # Fill in the total number of different weighted signals in use in the weighted tables
    # 'buy/sell__downwards/sideways/upwards_trend_total_signal_needed' settings will be multiplied with this value
    # so their search spaces will be larger, resulting in more equally divided weighted signal scores when hyperopting
    number_of_weighted_signals = 9

    # ROI Table StepSize:
    # Size of the steps in minutes to be used when calculating the long continuous ROI table
    # MGM generates a custom really long table so it will have less gaps in it and be more continuous in it's decrease
    roi_table_step_size = 5

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

    # Plot configuration to show all signals used in MoniGoMani in FreqUI (Use load from Strategy in FreqUI)
    plot_config = {
        'main_plot': {
            # Main Plot Indicators (SMAs, EMAs, Bollinger Bands, VWAP)
            'sma9': {'color': '#2c05f6'},
            'sma50': {'color': '#19038a'},
            'sma200': {'color': '#0d043b'},
            'ema9': {'color': '#12e5a6'},
            'ema50': {'color': '#0a8963'},
            'ema200': {'color': '#074b36'},
            'bb_upperband': {'color': '#6f1a7b'},
            'bb_lowerband': {'color': '#6f1a7b'},
            'vwap': {'color': '#727272'}
        },
        'subplots': {
            # Subplots - Each dict defines one additional plot (MACD, ADX, Plus/Minus Direction, RSI)
            'MACD (Moving Average Convergence Divergence)': {
                'macd': {'color': '#19038a'},
                'macdsignal': {'color': '#ae231c'}
            },
            'ADX (Average Directional Index) + Plus & Minus Directions': {
                'adx': {'color': '#6f1a7b'},
                'plus_di': {'color': '#0ad628'},
                'minus_di': {'color': '#ae231c'}
            },
            'RSI (Relative Strength Index)': {
                'rsi': {'color': '#7fba3c'}
            }
        }
    }

    # HyperOpt Settings Override
    # --------------------------
    # When the Parameters in below HyperOpt Space Parameters sections are altered as following examples then they can be
    # used as overrides while hyperopting / backtesting / dry/live-running (only truly useful when hyperopting though!)
    # Meaning you can use this to set individual buy_params/sell_params to a fixed value when hyperopting!
    # WARNING: Always double check that when doing a fresh hyperopt or doing a dry/live-run that all overrides are
    # turned off!
    #
    # Override Examples:
    # Override to False:    CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=False)
    # Override to 0:        IntParameter(0, int(100*precision), default=0, space='sell', optimize=False, load=False)
    #
    # default=           The value used when overriding
    # optimize=False     Exclude from hyperopting (Make static)
    # load=False         Don't load from above HYPEROPT RESULTS COPY-PASTE SECTION

    # ---------------------------------------------------------------- #
    #                  Buy HyperOpt Space Parameters                   #
    # ---------------------------------------------------------------- #

    # Trend Detecting Buy Signal Weight Influence Tables
    # -------------------------------------------------------
    # The idea is to let hyperopt find out which signals are more important over other signals by allocating weights to
    # them while also finding the "perfect" weight division between each-other.
    # These Signal Weight Influence Tables will be allocated to signals when their respective trend is detected
    # (Signals can be turned off by allocating 0 or turned into an override by setting them equal to or higher then
    # total_buy_signal_needed)

    # React to Buy Signals when certain trends are detected (False would disable trading in said trend)
    buy___trades_when_downwards = \
        CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=False)
    buy___trades_when_sideways = \
        CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=False)
    buy___trades_when_upwards = \
        CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=False)

    # Downwards Trend Buy
    # -------------------

    # Total Buy Signal Weight needed for Downwards Trends, calculated over a small lookback window, 
    # to check if an actual buy should occur
    buy__downwards_trend_total_signal_needed = \
        IntParameter(int(30 * precision), int(100 * number_of_weighted_signals * precision),
                     default=int(30 * precision), space='buy', optimize=True, load=True)
    buy__downwards_trend_total_signal_needed_candles_lookback_window = \
        IntParameter(1, 6, default=1, space='buy', optimize=True, load=True)

    # Buy Signal Weight Influence Table
    buy_downwards_trend_adx_strong_up_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_bollinger_bands_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_ema_long_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_ema_short_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_macd_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_rsi_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_sma_long_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_sma_short_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_vwap_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)

    # Sideways Trend Buy
    # ------------------

    # Total Buy Signal Weight needed for Sideways Trends, calculated over a small lookback window, 
    # to check if an actual buy should occur
    buy__sideways_trend_total_signal_needed = \
        IntParameter(int(30 * precision), int(100 * number_of_weighted_signals * precision),
                     default=int(30 * precision), space='buy', optimize=True, load=True)
    buy__sideways_trend_total_signal_needed_candles_lookback_window = \
        IntParameter(1, 6, default=1, space='buy', optimize=True, load=True)

    # Buy Signal Weight Influence Table
    buy_sideways_trend_adx_strong_up_weight = \
        IntParameter(56, int(76 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_bollinger_bands_weight = \
        IntParameter(46, int(66 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_ema_long_golden_cross_weight = \
        IntParameter(49, int(69 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_ema_short_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=100, space='buy', optimize=False, load=False)
    buy_sideways_trend_macd_weight = \
        IntParameter(59, int(79 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_rsi_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=False, load=False)
    buy_sideways_trend_sma_long_golden_cross_weight = \
        IntParameter(34, int(54 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_sma_short_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=100, space='buy', optimize=False, load=False)
    buy_sideways_trend_vwap_cross_weight = \
        IntParameter(62, int(82 * precision), default=0, space='buy', optimize=True, load=True)

    # Upwards Trend Buy
    # -----------------

    # Total Buy Signal Weight needed for Upwards Trends, calculated over a small lookback window, 
    # to check if an actual buy should occur
    buy__upwards_trend_total_signal_needed = \
        IntParameter(int(30 * precision), int(100 * number_of_weighted_signals * precision),
                     default=int(30 * precision), space='buy', optimize=True, load=True)
    buy__upwards_trend_total_signal_needed_candles_lookback_window = \
        IntParameter(1, 6, default=1, space='buy', optimize=True, load=True)

    # Buy Signal Weight Influence Table
    buy_upwards_trend_adx_strong_up_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_bollinger_bands_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_ema_long_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_ema_short_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_macd_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_rsi_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_sma_long_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_sma_short_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_vwap_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)

    # ---------------------------------------------------------------- #
    #                  Sell HyperOpt Space Parameters                  #
    # ---------------------------------------------------------------- #

    # Trend Detecting Buy Signal Weight Influence Tables
    # -------------------------------------------------------
    # The idea is to let hyperopt find out which signals are more important over other signals by allocating weights to
    # them while also finding the "perfect" weight division between each-other.
    # These Signal Weight Influence Tables will be allocated to signals when their respective trend is detected
    # (Signals can be turned off by allocating 0 or turned into an override by setting them equal to or higher then
    # total_buy_signal_needed)

    # React to Sell Signals when certain trends are detected (False would disable trading in said trend)
    sell___trades_when_downwards = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=False)
    sell___trades_when_sideways = \
        CategoricalParameter([True, False], default=False, space='sell', optimize=False, load=False)
    sell___trades_when_upwards = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=False)

    # Downwards Trend Sell
    # --------------------

    # Total Sell Signal Weight needed for Downwards Trends, calculated over a small lookback window, 
    # to check if an actual sell should occur
    sell__downwards_trend_total_signal_needed = \
        IntParameter(int(30 * precision), int(100 * number_of_weighted_signals * precision),
                     default=int(30 * precision), space='sell', optimize=True, load=True)
    sell__downwards_trend_total_signal_needed_candles_lookback_window = \
        IntParameter(1, 6, default=1, space='sell', optimize=True, load=True)

    # Sell Signal Weight Influence Table
    sell_downwards_trend_adx_strong_down_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_bollinger_bands_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_ema_long_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_ema_short_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_macd_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_rsi_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_sma_long_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_sma_short_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_vwap_cross_weight = \
        IntParameter(5, int(100 * precision), default=0, space='sell', optimize=True, load=True)

    # Sideways Trend Sell
    # -------------------

    # Total Sell Signal Weight needed for Sideways Trends, calculated over a small lookback window, 
    # to check if an actual sell should occur
    sell__sideways_trend_total_signal_needed = \
        IntParameter(int(30 * precision), int(100 * number_of_weighted_signals * precision),
                     default=int(30 * precision), space='sell', optimize=True, load=True)
    sell__sideways_trend_total_signal_needed_candles_lookback_window = \
        IntParameter(1, 6, default=1, space='sell', optimize=True, load=True)

    # Sell Signal Weight Influence Table
    sell_sideways_trend_adx_strong_down_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_bollinger_bands_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_ema_long_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_ema_short_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_macd_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_rsi_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_sma_long_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_sma_short_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_vwap_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)

    # Upwards Trend Sell
    # ------------------

    # Total Sell Signal Weight needed for Sideways Trends, calculated over a small lookback window, 
    # to check if an actual sell should occur
    sell__upwards_trend_total_signal_needed = \
        IntParameter(int(30 * precision), int(100 * number_of_weighted_signals * precision),
                     default=int(30 * precision), space='sell', optimize=True, load=True)
    sell__upwards_trend_total_signal_needed_candles_lookback_window = \
        IntParameter(1, 6, default=1, space='sell', optimize=True, load=True)

    # Sell Signal Weight Influence Table
    sell_upwards_trend_adx_strong_down_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_bollinger_bands_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_ema_long_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_ema_short_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_macd_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_rsi_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_sma_long_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_sma_short_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_vwap_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)

    # ---------------------------------------------------------------- #
    #             Sell Unclogger HyperOpt Space Parameters             #
    # ---------------------------------------------------------------- #

    sell___unclogger_enabled = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=False)
    sell___unclogger_minimal_losing_trade_duration_minutes = \
        IntParameter(int(15 * precision), int(60 * precision), default=int(15 * precision), space='sell', optimize=True, load=True)
    sell___unclogger_minimal_losing_trades_open = \
        IntParameter(1, 5, default=1, space='sell', optimize=True, load=True)
    sell___unclogger_open_trades_losing_percentage_needed = \
        IntParameter(1, int(60 * precision), default=1, space='sell', optimize=True, load=True)
    sell___unclogger_trend_lookback_candles_window = \
        IntParameter(int(10 * precision), int(60 * precision), default=int(10 * precision), space='sell', optimize=True, load=True)
    sell___unclogger_trend_lookback_candles_window_percentage_needed = \
        IntParameter(int(10 * precision), int(40 * precision), default=int(10 * precision), space='sell', optimize=True, load=True)
    sell___unclogger_trend_lookback_window_uses_downwards_candles = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=False)
    sell___unclogger_trend_lookback_window_uses_sideways_candles = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=False)
    sell___unclogger_trend_lookback_window_uses_upwards_candles = \
        CategoricalParameter([True, False], default=False, space='sell', optimize=False, load=False)

    class HyperOpt:
        # Generate a Custom Long Continuous ROI-Table with less gaps in it
        @staticmethod
        def generate_roi_table(params):
            step = MoniGoManiHyperStrategy.roi_table_step_size
            minimal_roi = {0: params['roi_p1'] + params['roi_p2'] + params['roi_p3'],
                           params['roi_t3']: params['roi_p1'] + params['roi_p2'],
                           params['roi_t3'] + params['roi_t2']: params['roi_p1'],
                           params['roi_t3'] + params['roi_t2'] + params['roi_t1']: 0}

            max_value = max(map(int, minimal_roi.keys()))
            f = interp1d(
                list(map(int, minimal_roi.keys())),
                list(minimal_roi.values())
            )
            x = list(range(0, max_value, step))
            y = list(map(float, map(f, x)))
            if y[-1] != 0:
                x.append(x[-1] + step)
                y.append(0)
            return dict(zip(x, y))

    def __init__(self, config: dict):
        """
        First method to be called once during the MoniGoMani class initialization process
        :param config::
        """

        super().__init__(config)
        initialization = 'Initialization'

        if RunMode(config.get('runmode', RunMode.OTHER)) in (RunMode.BACKTEST, RunMode.HYPEROPT):
            self.timeframe = self.backtest_timeframe
            self.mgm_logger('info', 'TimeFrame-Zoom', f'Auto updating to zoomed "backtest_timeframe": {self.timeframe}')

            self.is_dry_live_run_detected = False
            self.mgm_logger('info', initialization, f'Current run mode detected as: HyperOpting/BackTesting. '
                                                    f'Auto updated is_dry_live_run_detected to: False')
        else:
            self.is_dry_live_run_detected = True
            self.mgm_logger('info', initialization, f'Current run mode detected as: Dry/Live-Run. '
                                                    f'Auto updated is_dry_live_run_detected to: True')

    def informative_pairs(self):
        """
        Defines additional informative pair/interval combinations to be cached from the exchange, these will be used
        during TimeFrame-Zoom.
        :return:
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds indicators based on Run-Mode & TimeFrame-Zoom:

        If Dry/Live-running or BackTesting/HyperOpting without TimeFrame-Zoom it just pulls 'timeframe' (1h candles) to
        compute indicators.

        If BackTesting/HyperOpting with TimeFrame-Zoom it pulls 'informative_pairs' (1h candles) to compute indicators,
        but then tests upon 'backtest_timeframe' (5m or 1m candles) to simulate price movement during that 'timeframe'
        (1h candle).

        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for MoniGoMani
        """
        timeframe_zoom = 'TimeFrame-Zoom'
        # Compute indicator data during Backtesting / Hyperopting when TimeFrame-Zooming
        if (self.is_dry_live_run_detected is False) and (self.informative_timeframe != self.backtest_timeframe):
            self.mgm_logger('info', timeframe_zoom, f'Backtesting/Hyperopting this strategy with a '
                                                    f'informative_timeframe ({self.informative_timeframe} candles) and '
                                                    f'a zoomed backtest_timeframe ({self.backtest_timeframe} candles)')

            # Warning! This method gets ALL downloaded data that you have (when in backtesting mode).
            # If you have many months or years downloaded for this pair, this will take a long time!
            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

            # Throw away older data that isn't needed.
            first_informative = dataframe["date"].min().floor("H")
            informative = informative[informative["date"] >= first_informative]

            # Populate indicators at a larger timeframe
            informative = self._populate_indicators(informative.copy(), metadata)

            # Merge indicators back in with, filling in missing values.
            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe,
                                               ffill=True)

            # Rename columns, since merge_informative_pair adds `_<timeframe>` to the end of each name.
            # Skip over date etc..
            skip_columns = [(s + "_" + self.informative_timeframe) for s in
                            ['date', 'open', 'high', 'low', 'close', 'volume']]
            dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if
            (not s in skip_columns) else s, inplace=True)

        # Compute indicator data normally during Dry & Live Running or when not using TimeFrame-Zoom
        else:
            self.mgm_logger('info', timeframe_zoom,
                            f'Dry/Live-running MoniGoMani with normal timeframe ({self.timeframe} candles)')
            # Just populate indicators.
            dataframe = self._populate_indicators(dataframe, metadata)

        return dataframe

    def _populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame.
        Should be called with 'informative_pair' (1h candles) during backtesting/hyperopting with TimeFrame-Zoom!

        Performance Note: For the best performance be frugal on the number of indicators you are using.
        Let uncomment only the indicator you are using in MoniGoMani or your hyperopt configuration,
        otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for MoniGoMani
        """

        # Momentum Indicators (timeperiod is expressed in candles)
        # -------------------

        # ADX - Average Directional Index (The Trend Strength Indicator)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)  # 14 timeperiods is usually used for ADX

        # +DM (Positive Directional Indicator) = current high - previous high
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=25)
        # -DM (Negative Directional Indicator) = previous low - current low
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=25)

        # RSI - Relative Strength Index (Under bought / Over sold & Over bought / Under sold indicator Indicator)
        dataframe['rsi'] = ta.RSI(dataframe)

        # MACD - Moving Average Convergence Divergence
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']  # MACD - Blue TradingView Line (Bullish if on top)
        dataframe['macdsignal'] = macd['macdsignal']  # Signal - Orange TradingView Line (Bearish if on top)

        # Overlap Studies
        # ---------------

        # SMA's & EMA's are trend following tools (Should not be used when line goes sideways)
        # SMA - Simple Moving Average (Moves slower compared to EMA, price trend over X periods)
        dataframe['sma9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)

        # EMA - Exponential Moving Average (Moves quicker compared to SMA, more weight added)
        # (For traders who trade intra-day and fast-moving markets, the EMA is more applicable)
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)  # timeperiod is expressed in candles
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']

        # Volume Indicators
        # -----------------

        # VWAP - Volume Weighted Average Price
        dataframe['vwap'] = qtpylib.vwap(dataframe)

        # Weighted Variables
        # ------------------

        # Initialize weighted buy/sell signal variables if they are needed (should be 0 = false by default)
        if self.debuggable_weighted_signal_dataframe:
            dataframe['adx_strong_up_weighted_buy_signal'] = dataframe['adx_strong_down_weighted_sell_signal'] = 0
            dataframe['bollinger_bands_weighted_buy_signal'] = dataframe['bollinger_bands_weighted_sell_signal'] = 0
            dataframe['ema_long_death_cross_weighted_sell_signal'] = 0
            dataframe['ema_long_golden_cross_weighted_buy_signal'] = 0
            dataframe['ema_short_death_cross_weighted_sell_signal'] = 0
            dataframe['ema_short_golden_cross_weighted_buy_signal'] = 0
            dataframe['macd_weighted_buy_signal'] = dataframe['macd_weighted_sell_signal'] = 0
            dataframe['rsi_weighted_buy_signal'] = dataframe['rsi_weighted_sell_signal'] = 0
            dataframe['sma_long_death_cross_weighted_sell_signal'] = 0
            dataframe['sma_long_golden_cross_weighted_buy_signal'] = 0
            dataframe['sma_short_death_cross_weighted_sell_signal'] = 0
            dataframe['sma_short_golden_cross_weighted_buy_signal'] = 0
            dataframe['vwap_cross_weighted_buy_signal'] = dataframe['vwap_cross_weighted_sell_signal'] = 0

        # Initialize total signal variables (should be 0 = false by default)
        dataframe['total_buy_signal_strength'] = dataframe['total_sell_signal_strength'] = 0

        # Trend Detection
        # ---------------

        # Detect if current trend going Downwards / Sideways / Upwards, strategy will respond accordingly
        dataframe.loc[(dataframe['adx'] > 22) & (dataframe['plus_di'] < dataframe['minus_di']), 'trend'] = 'downwards'
        dataframe.loc[dataframe['adx'] < 22, 'trend'] = 'sideways'
        dataframe.loc[(dataframe['adx'] > 22) & (dataframe['plus_di'] > dataframe['minus_di']), 'trend'] = 'upwards'

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        # If a Weighted Buy Signal goes off => Bullish Indication, Set to true (=1) and multiply by weight percentage

        if self.debuggable_weighted_signal_dataframe:
            # Weighted Buy Signal: ADX above 25 & +DI above -DI (The trend has strength while moving up)
            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['adx'] > 25),
                          'adx_strong_up_weighted_buy_signal'] = \
                self.buy_downwards_trend_adx_strong_up_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['adx'] > 25),
                          'adx_strong_up_weighted_buy_signal'] = \
                self.buy_sideways_trend_adx_strong_up_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['adx'] > 25),
                          'adx_strong_up_weighted_buy_signal'] = \
                self.buy_upwards_trend_adx_strong_up_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['adx_strong_up_weighted_buy_signal']

            # Weighted Buy Signal: Re-Entering Lower Bollinger Band after downward breakout
            # (Candle closes below Upper Bollinger Band)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'bollinger_bands_weighted_buy_signal'] = \
                self.buy_downwards_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'bollinger_bands_weighted_buy_signal'] = \
                self.buy_sideways_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'bollinger_bands_weighted_buy_signal'] = \
                self.buy_upwards_trend_bollinger_bands_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['bollinger_bands_weighted_buy_signal']

            # Weighted Buy Signal: EMA long term Golden Cross (Medium term EMA crosses above Long term EMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_golden_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_ema_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_golden_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_ema_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_golden_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_ema_long_golden_cross_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['ema_long_golden_cross_weighted_buy_signal']

            # Weighted Buy Signal: EMA short term Golden Cross (Short term EMA crosses above Medium term EMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_golden_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_ema_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_golden_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_ema_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_golden_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_ema_short_golden_cross_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['ema_short_golden_cross_weighted_buy_signal']

            # Weighted Buy Signal: MACD above Signal
            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['macd'] > dataframe['macdsignal']),
                          'macd_weighted_buy_signal'] = self.buy_downwards_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['macd'] > dataframe['macdsignal']),
                          'macd_weighted_buy_signal'] = self.buy_sideways_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['macd'] > dataframe['macdsignal']),
                          'macd_weighted_buy_signal'] = self.buy_upwards_trend_macd_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['macd_weighted_buy_signal']

            # Weighted Buy Signal: RSI crosses above 30 (Under-bought / low-price and rising indication)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'rsi_weighted_buy_signal'] = self.buy_downwards_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'rsi_weighted_buy_signal'] = self.buy_sideways_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'rsi_weighted_buy_signal'] = self.buy_upwards_trend_rsi_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['rsi_weighted_buy_signal']

            # Weighted Buy Signal: SMA long term Golden Cross (Medium term SMA crosses above Long term SMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_golden_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_sma_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_golden_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_sma_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_golden_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_sma_long_golden_cross_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['sma_long_golden_cross_weighted_buy_signal']

            # Weighted Buy Signal: SMA short term Golden Cross (Short term SMA crosses above Medium term SMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_golden_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_sma_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_golden_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_sma_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_golden_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_sma_short_golden_cross_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['sma_short_golden_cross_weighted_buy_signal']

            # Weighted Buy Signal: VWAP crosses above current price (Simultaneous rapid increase in volume and price)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_vwap_cross_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['vwap_cross_weighted_buy_signal']

        else:
            # Weighted Buy Signal: ADX above 25 & +DI above -DI (The trend has strength while moving up)
            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['adx'] > 25),
                          'total_buy_signal_strength'] += \
                self.buy_downwards_trend_adx_strong_up_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['adx'] > 25),
                          'total_buy_signal_strength'] += \
                self.buy_sideways_trend_adx_strong_up_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['adx'] > 25),
                          'total_buy_signal_strength'] += \
                self.buy_upwards_trend_adx_strong_up_weight.value / self.precision

            # Weighted Buy Signal: Re-Entering Lower Bollinger Band after downward breakout
            # (Candle closes below Upper Bollinger Band)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'total_buy_signal_strength'] += \
                self.buy_downwards_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'total_buy_signal_strength'] += \
                self.buy_sideways_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'total_buy_signal_strength'] += \
                self.buy_upwards_trend_bollinger_bands_weight.value / self.precision

            # Weighted Buy Signal: EMA long term Golden Cross (Medium term EMA crosses above Long term EMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'total_buy_signal_strength'] += \
                self.buy_downwards_trend_ema_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'total_buy_signal_strength'] += \
                self.buy_sideways_trend_ema_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'total_buy_signal_strength'] += \
                self.buy_upwards_trend_ema_long_golden_cross_weight.value / self.precision

            # Weighted Buy Signal: EMA short term Golden Cross (Short term EMA crosses above Medium term EMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'total_buy_signal_strength'] += \
                self.buy_downwards_trend_ema_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'total_buy_signal_strength'] += \
                self.buy_sideways_trend_ema_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'total_buy_signal_strength'] += \
                self.buy_upwards_trend_ema_short_golden_cross_weight.value / self.precision

            # Weighted Buy Signal: MACD above Signal
            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['macd'] > dataframe['macdsignal']),
                          'total_buy_signal_strength'] += self.buy_downwards_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['macd'] > dataframe['macdsignal']),
                          'total_buy_signal_strength'] += self.buy_sideways_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['macd'] > dataframe['macdsignal']),
                          'total_buy_signal_strength'] += self.buy_upwards_trend_macd_weight.value / self.precision

            # Weighted Buy Signal: RSI crosses above 30 (Under-bought / low-price and rising indication)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'total_buy_signal_strength'] += self.buy_downwards_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'total_buy_signal_strength'] += self.buy_sideways_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'total_buy_signal_strength'] += self.buy_upwards_trend_rsi_weight.value / self.precision

            # Weighted Buy Signal: SMA long term Golden Cross (Medium term SMA crosses above Long term SMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'total_buy_signal_strength'] += \
                self.buy_downwards_trend_sma_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'total_buy_signal_strength'] += \
                self.buy_sideways_trend_sma_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'total_buy_signal_strength'] += \
                self.buy_upwards_trend_sma_long_golden_cross_weight.value / self.precision

            # Weighted Buy Signal: SMA short term Golden Cross (Short term SMA crosses above Medium term SMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'total_buy_signal_strength'] += \
                self.buy_downwards_trend_sma_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'total_buy_signal_strength'] += \
                self.buy_sideways_trend_sma_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'total_buy_signal_strength'] += \
                self.buy_upwards_trend_sma_short_golden_cross_weight.value / self.precision

            # Weighted Buy Signal: VWAP crosses above current price (Simultaneous rapid increase in volume and price)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'total_buy_signal_strength'] += \
                self.buy_downwards_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'total_buy_signal_strength'] += \
                self.buy_sideways_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'total_buy_signal_strength'] += \
                self.buy_upwards_trend_vwap_cross_weight.value / self.precision

        # Check if buy signal should be sent depending on the current trend, using a lookback window to take signals
        # that fired during previous candles into consideration
        dataframe.loc[
            (
                    (dataframe['trend'] == 'downwards') &
                    (dataframe['total_buy_signal_strength']
                     .rolling(self.buy__downwards_trend_total_signal_needed_candles_lookback_window.value).sum()
                     >= self.buy__downwards_trend_total_signal_needed.value / self.precision)
            ) | (
                    (dataframe['trend'] == 'sideways') &
                    (dataframe['total_buy_signal_strength']
                     .rolling(self.buy__sideways_trend_total_signal_needed_candles_lookback_window.value).sum()
                     >= self.buy__sideways_trend_total_signal_needed.value / self.precision)
            ) | (
                    (dataframe['trend'] == 'upwards') &
                    (dataframe['total_buy_signal_strength']
                     .rolling(self.buy__upwards_trend_total_signal_needed_candles_lookback_window.value).sum()
                     >= self.buy__upwards_trend_total_signal_needed.value / self.precision)
            ), 'buy'] = 1

        # Override Buy Signal: When configured buy signals can be completely turned off for each kind of trend
        if not self.buy___trades_when_downwards.value / self.precision:
            dataframe.loc[dataframe['trend'] == 'downwards', 'buy'] = 0
        if not self.buy___trades_when_sideways.value / self.precision:
            dataframe.loc[dataframe['trend'] == 'sideways', 'buy'] = 0
        if not self.buy___trades_when_upwards.value / self.precision:
            dataframe.loc[dataframe['trend'] == 'upwards', 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        # If a Weighted Sell Signal goes off => Bearish Indication, Set to true (=1) and multiply by weight percentage

        if self.debuggable_weighted_signal_dataframe:
            # Weighted Sell Signal: ADX above 25 & +DI below -DI (The trend has strength while moving down)
            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['adx'] > 25),
                          'adx_strong_down_weighted_sell_signal'] = \
                self.sell_downwards_trend_adx_strong_down_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['adx'] > 25),
                          'adx_strong_down_weighted_sell_signal'] = \
                self.sell_sideways_trend_adx_strong_down_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['adx'] > 25),
                          'adx_strong_down_weighted_sell_signal'] = \
                self.sell_upwards_trend_adx_strong_down_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['adx_strong_down_weighted_sell_signal']

            # Weighted Sell Signal: Re-Entering Upper Bollinger Band after upward breakout
            # (Candle closes below Upper Bollinger Band)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'bollinger_bands_weighted_sell_signal'] = \
                self.sell_downwards_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'bollinger_bands_weighted_sell_signal'] = \
                self.sell_sideways_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'bollinger_bands_weighted_sell_signal'] = \
                self.sell_upwards_trend_bollinger_bands_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['bollinger_bands_weighted_sell_signal']

            # Weighted Sell Signal: EMA long term Death Cross (Medium term EMA crosses below Long term EMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_death_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_ema_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_death_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_ema_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_death_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_ema_long_death_cross_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['ema_long_death_cross_weighted_sell_signal']

            # Weighted Sell Signal: EMA short term Death Cross (Short term EMA crosses below Medium term EMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_death_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_ema_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_death_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_ema_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_death_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_ema_short_death_cross_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['ema_short_death_cross_weighted_sell_signal']

            # Weighted Sell Signal: MACD below Signal
            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['macd'] < dataframe['macdsignal']),
                          'macd_weighted_sell_signal'] = self.sell_downwards_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['macd'] < dataframe['macdsignal']),
                          'macd_weighted_sell_signal'] = self.sell_sideways_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['macd'] < dataframe['macdsignal']),
                          'macd_weighted_sell_signal'] = self.sell_upwards_trend_macd_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['macd_weighted_sell_signal']

            # Weighted Sell Signal: RSI crosses below 70 (Over-bought / high-price and dropping indication)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'rsi_weighted_sell_signal'] = self.sell_downwards_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'rsi_weighted_sell_signal'] = self.sell_sideways_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'rsi_weighted_sell_signal'] = self.sell_upwards_trend_rsi_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['rsi_weighted_sell_signal']

            # Weighted Sell Signal: SMA long term Death Cross (Medium term SMA crosses below Long term SMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_death_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_sma_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_death_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_sma_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_death_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_sma_long_death_cross_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['sma_long_death_cross_weighted_sell_signal']

            # Weighted Sell Signal: SMA short term Death Cross (Short term SMA crosses below Medium term SMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_death_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_sma_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_death_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_sma_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_death_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_sma_short_death_cross_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['sma_short_death_cross_weighted_sell_signal']

            # Weighted Sell Signal: VWAP crosses below current price
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_vwap_cross_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['vwap_cross_weighted_sell_signal']

        else:
            # Weighted Sell Signal: ADX above 25 & +DI below -DI (The trend has strength while moving down)
            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['adx'] > 25),
                          'total_sell_signal_strength'] += \
                self.sell_downwards_trend_adx_strong_down_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['adx'] > 25),
                          'total_sell_signal_strength'] += \
                self.sell_sideways_trend_adx_strong_down_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['adx'] > 25),
                          'total_sell_signal_strength'] += \
                self.sell_upwards_trend_adx_strong_down_weight.value / self.precision

            # Weighted Sell Signal: Re-Entering Upper Bollinger Band after upward breakout
            # (Candle closes below Upper Bollinger Band)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_bollinger_bands_weight.value / self.precision

            # Weighted Sell Signal: EMA long term Death Cross (Medium term EMA crosses below Long term EMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_ema_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_ema_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_ema_long_death_cross_weight.value / self.precision

            # Weighted Sell Signal: EMA short term Death Cross (Short term EMA crosses below Medium term EMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_ema_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_ema_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_ema_short_death_cross_weight.value / self.precision

            # Weighted Sell Signal: MACD below Signal
            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['macd'] < dataframe['macdsignal']),
                          'total_sell_signal_strength'] += self.sell_downwards_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['macd'] < dataframe['macdsignal']),
                          'total_sell_signal_strength'] += self.sell_sideways_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['macd'] < dataframe['macdsignal']),
                          'total_sell_signal_strength'] += self.sell_upwards_trend_macd_weight.value / self.precision

            # Weighted Sell Signal: RSI crosses below 70 (Over-bought / high-price and dropping indication)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'total_sell_signal_strength'] += self.sell_downwards_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'total_sell_signal_strength'] += self.sell_sideways_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'total_sell_signal_strength'] += self.sell_upwards_trend_rsi_weight.value / self.precision

            # Weighted Sell Signal: SMA long term Death Cross (Medium term SMA crosses below Long term SMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_sma_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_sma_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_sma_long_death_cross_weight.value / self.precision

            # Weighted Sell Signal: SMA short term Death Cross (Short term SMA crosses below Medium term SMA)
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_sma_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_sma_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_sma_short_death_cross_weight.value / self.precision

            # Weighted Sell Signal: VWAP crosses below current price
            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_vwap_cross_weight.value / self.precision

        # Check if buy signal should be sent depending on the current trend, using a lookback window to take signals
        # that fired during previous candles into consideration
        dataframe.loc[
            (
                    (dataframe['trend'] == 'downwards') &
                    (dataframe['total_sell_signal_strength']
                     .rolling(self.sell__downwards_trend_total_signal_needed_candles_lookback_window.value).sum()
                     >= self.sell__downwards_trend_total_signal_needed.value / self.precision)
            ) | (
                    (dataframe['trend'] == 'sideways') &
                    (dataframe['total_sell_signal_strength']
                     .rolling(self.sell__sideways_trend_total_signal_needed_candles_lookback_window.value).sum()
                     >= self.sell__sideways_trend_total_signal_needed.value / self.precision)
            ) | (
                    (dataframe['trend'] == 'upwards') &
                    (dataframe['total_sell_signal_strength']
                     .rolling(self.sell__upwards_trend_total_signal_needed_candles_lookback_window.value).sum()
                     >= self.sell__upwards_trend_total_signal_needed.value / self.precision)
            ), 'sell'] = 1

        # Override Sell Signal: When configured sell signals can be completely turned off for each kind of trend
        if not self.sell___trades_when_downwards.value / self.precision:
            dataframe.loc[dataframe['trend'] == 'downwards', 'sell'] = 0
        if not self.sell___trades_when_sideways.value / self.precision:
            dataframe.loc[dataframe['trend'] == 'sideways', 'sell'] = 0
        if not self.sell___trades_when_upwards.value / self.precision:
            dataframe.loc[dataframe['trend'] == 'upwards', 'sell'] = 0

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Open Trade Custom Information Storage & Garbage Collector
        ---------------------------------------------------------
        MoniGoMani (currently) only uses this function to store custom information from all open_trades at that given
        moment during BackTesting/HyperOpting or Dry/Live-Running
        Further it also does garbage collection to make sure no old closed trade data remains in custom_info

        The actual normal "custom_stoploss" usage for which this function is generally used isn't used by MGM (yet)!
        This custom_stoploss function should be able to work in tandem with Trailing stoploss!

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break MoniGoMani.
        :return float: New stoploss value, relative to the current-rate
        """

        custom_information_storage = 'custom_stoploss - Custom Information Storage'
        garbage_collector = custom_information_storage + ' Garbage Collector'

        # Open Trade Custom Information Storage
        # -------------------------------------
        self.mgm_logger('debug', custom_information_storage, f'Fetching all currently open trades')

        # Fetch all open trade data during Dry & Live Running
        if self.is_dry_live_run_detected is True:
            self.mgm_logger('debug', custom_information_storage,
                            f'Fetching all currently open trades during Dry/Live Run')

            all_open_trades = Trade.get_trades([Trade.is_open.is_(True)]).order_by(Trade.open_date).all()
        # Fetch all open trade data during Back Testing & Hyper Opting
        else:
            self.mgm_logger('debug', custom_information_storage,
                            f'Fetching all currently open trades during BackTesting/HyperOpting')
            all_open_trades = trade.trades_open

        self.mgm_logger('debug', custom_information_storage,
                        f'Up-to-date open trades ({str(len(all_open_trades))}) fetched!')
        self.mgm_logger('debug', custom_information_storage,
                        f'all_open_trades contents: {repr(all_open_trades)}')

        # Store current pair's open_trade + it's current profit in custom_info
        for open_trade in all_open_trades:
            if str(open_trade.pair) == str(pair):
                if str(open_trade.pair) not in self.custom_info['open_trades']:
                    self.custom_info['open_trades'][str(open_trade.pair)] = {}
                self.custom_info['open_trades'][str(open_trade.pair)]['trade'] = str(open_trade)
                self.custom_info['open_trades'][str(open_trade.pair)]['current_profit'] = current_profit
                self.mgm_logger('info', custom_information_storage,
                                f'Storing trade + current profit/loss for pair ({str(pair)}) '
                                f'in custom_info')
                break

        # Custom Information Storage Garbage Collector
        # --------------------------------------------
        # Check if any old open_trade garbage needs to be removed
        if len(all_open_trades) < len(self.custom_info['open_trades']):
            garbage_trade_amount = len(self.custom_info['open_trades']) - len(all_open_trades)
            self.mgm_logger('info', garbage_collector, f'Old open trade garbage detected for '
                                                       f'{str(garbage_trade_amount)} trades, starting cleanup')

            for garbage_trade in range(garbage_trade_amount):
                for stored_trade in self.custom_info['open_trades']:
                    pair_still_open = False
                    for open_trade in all_open_trades:
                        if str(stored_trade) == str(open_trade.pair):
                            self.mgm_logger('debug', garbage_collector,
                                            f'Open trade found, no action needed for pair ({stored_trade}) '
                                            f'in custom_info')
                            pair_still_open = True
                            break

                    # Remove old open_trade garbage
                    if not pair_still_open:
                        self.mgm_logger('info', garbage_collector,
                                        f'No open trade found for pair ({stored_trade}), removing '
                                        f'from custom_info')
                        self.custom_info['open_trades'].pop(stored_trade)
                        self.mgm_logger('debug', garbage_collector,
                                        f'Successfully removed garbage_trade {str(garbage_trade)} '
                                        f'from custom_info!')
                        break

        # Print all stored open trade info in custom_storage
        self.mgm_logger('debug', custom_information_storage,
                        f'Open trades ({str(len(self.custom_info["open_trades"]))}) in custom_info updated '
                        f'successfully!')
        self.mgm_logger('debug', custom_information_storage,
                        f'custom_info["open_trades"] contents: {repr(self.custom_info["open_trades"])}')

        # Always return a value bigger than the initial stoploss to keep using the initial stoploss.
        # Since we (currently) only want to use this function for custom information storage!
        return -1

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        """
        Open Trade Unclogger:
        ---------------------
        Override Sell Signal: When enabled attempts to unclog the bot when it's stuck with losing trades & unable to
        trade more new trades.

        It will only unclog a losing trade when all of following checks have been full-filled:
        => Check if everything in custom_storage is up to date with all_open_trades
        => Check if there are enough losing trades open for unclogging to occur
        => Check if there is a losing trade open for the pair currently being ran through the MoniGoMani loop
        => Check if trade has been open for X minutes (long enough to give it a recovery chance)
        => Check if total open trades losing % is met
        => Check if open_trade's trend changed negatively during past X candles

        Please configurable/hyperoptable in the sell_params dictionary under the hyperopt results copy/paste section.
        Only used when sell_params['sell___unclogger_enabled'] is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break MoniGoMani.
        :return float: New stoploss value, relative to the current-rate
        """

        open_trade_unclogger = 'Open Trade Unclogger'
        custom_information_storage = 'custom_sell - Custom Information Storage'

        if self.sell___unclogger_enabled.value:
            try:
                # Open Trade Custom Information Storage
                # -------------------------------------
                # Fetch all open trade data during Dry & Live Running
                if self.is_dry_live_run_detected is True:
                    self.mgm_logger('debug', custom_information_storage,
                                    f'Fetching all currently open trades during Dry/Live Run')

                    all_open_trades = Trade.get_trades([Trade.is_open.is_(True)]).order_by(Trade.open_date).all()
                # Fetch all open trade data during Back Testing & Hyper Opting
                else:
                    self.mgm_logger('debug', custom_information_storage,
                                    f'Fetching all currently open trades during BackTesting/HyperOpting')
                    all_open_trades = trade.trades_open

                self.mgm_logger('debug', custom_information_storage,
                                f'Up-to-date open trades ({str(len(all_open_trades))}) fetched!')
                self.mgm_logger('debug', custom_information_storage,
                                f'all_open_trades contents: {repr(all_open_trades)}')

                # Check if everything in custom_storage is up to date with all_open_trades
                if len(all_open_trades) > len(self.custom_info['open_trades']):
                    self.mgm_logger('warning', custom_information_storage,
                                    f'Open trades ({str(len(self.custom_info["open_trades"]))}) in custom_storage do '
                                    f'not match yet with trades in live open trades ({str(len(all_open_trades))}) '
                                    f'aborting unclogger for now!')
                else:
                    # Open Trade Unclogger
                    # --------------------
                    self.mgm_logger('debug', open_trade_unclogger,
                                    f'Running trough all checks to see if unclogging is needed')

                    # Check if there are enough losing trades open for unclogging to occur
                    self.mgm_logger('debug', open_trade_unclogger,
                                    f'Fetching all currently losing_open_trades from custom information storage')
                    losing_open_trades = {}
                    for stored_trade in self.custom_info['open_trades']:
                        stored_current_profit = self.custom_info['open_trades'][stored_trade]['current_profit']
                        if stored_current_profit < 0:
                            if not str(pair) in losing_open_trades:
                                losing_open_trades[str(stored_trade)] = {}
                            losing_open_trades[str(stored_trade)] = stored_current_profit
                    self.mgm_logger('debug', open_trade_unclogger,
                                    f'Fetched losing_open_trades ({str(len(losing_open_trades))}) from custom '
                                    f'information storage!')

                    if len(losing_open_trades) < self.sell___unclogger_minimal_losing_trades_open.value:
                        self.mgm_logger('debug', open_trade_unclogger,
                                        f'No unclogging needed! Not enough losing trades currently open!')
                    else:
                        self.mgm_logger('debug', open_trade_unclogger,
                                        f'Enough losing trades detected! Proceeding to the next check!')

                        # Check if there is a losing trade open for the pair currently being ran through the MoniGoMani
                        if pair not in losing_open_trades:
                            self.mgm_logger('debug', open_trade_unclogger,
                                            f'No unclogging needed! Currently checked pair ({pair}) is not making a '
                                            f'loss at this point in time!')
                        else:
                            self.mgm_logger('debug', open_trade_unclogger,
                                            f'Currently checked pair ({pair}) is losing! Proceeding to the next check!')

                            self.mgm_logger('debug', open_trade_unclogger,
                                            f'Trade open time: {str(trade.open_date_utc.replace(tzinfo=None))}')

                            minimal_open_time = current_time.replace(tzinfo=None) - timedelta(minutes=round(
                                self.sell___unclogger_minimal_losing_trade_duration_minutes.value / self.precision))

                            self.mgm_logger('debug', open_trade_unclogger,
                                            f'Minimal open time: {str(minimal_open_time)}')

                            if trade.open_date_utc.replace(tzinfo=None) > minimal_open_time:
                                self.mgm_logger('debug', open_trade_unclogger,
                                                f'No unclogging needed! Currently checked pair ({pair}) has not been '
                                                f'open been open for long enough!')
                            else:
                                self.mgm_logger('debug', open_trade_unclogger,
                                                f'Trade has been open for long enough! Proceeding to the next check!')

                                # Check if total open trades losing % is met
                                percentage_open_trades_losing = \
                                    int((len(losing_open_trades) / len(all_open_trades)) * 100)
                                self.mgm_logger('debug', open_trade_unclogger,
                                                f'percentage_open_trades_losing: {str(percentage_open_trades_losing)}%')
                                if percentage_open_trades_losing < \
                                        round(self.sell___unclogger_open_trades_losing_percentage_needed.value /
                                              self.precision):
                                    self.mgm_logger('debug', open_trade_unclogger,
                                                    f'No unclogging needed! Percentage of open trades losing needed '
                                                    f'has not been satisfied!')
                                else:
                                    self.mgm_logger('debug', open_trade_unclogger,
                                                    f'Percentage of open trades losing needed has been satisfied! '
                                                    f'Proceeding to the next check!')

                                    # Fetch current dataframe for the pair currently being ran through MoniGoMani
                                    self.mgm_logger('debug', open_trade_unclogger,
                                                    f'Fetching currently needed "trend" dataframe data to check how '
                                                    f'pair ({pair}) has been doing in during the last '
                                                    f'{str(self.sell___unclogger_trend_lookback_candles_window.value / self.precision)}'
                                                    f' candles')

                                    # Fetch all needed 'trend' trade data
                                    stored_trend_dataframe = {}
                                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

                                    self.mgm_logger('debug', open_trade_unclogger,
                                                    f'Fetching all needed "trend" trade data')

                                    for candle in range(1, round(self.sell___unclogger_trend_lookback_candles_window.value / self.precision) + 1):
                                        # Convert the candle time to the one being used by the
                                        # 'informative_timeframe'
                                        candle_multiplier = int(self.informative_timeframe.rstrip("mhdwM"))
                                        candle_time = \
                                            timeframe_to_prev_date(self.informative_timeframe, current_time) - \
                                            timedelta(minutes=int(candle * candle_multiplier))
                                        if self.informative_timeframe.find('h') != -1:
                                            candle_time = \
                                                timeframe_to_prev_date(self.informative_timeframe, current_time) - \
                                                timedelta(hours=int(candle * candle_multiplier))
                                        elif self.informative_timeframe.find('d') != -1:
                                            candle_time =\
                                                timeframe_to_prev_date(self.informative_timeframe, current_time) - \
                                                timedelta(days=int(candle * candle_multiplier))
                                        elif self.informative_timeframe.find('w') != -1:
                                            candle_time = \
                                                timeframe_to_prev_date(self.informative_timeframe, current_time) - \
                                                timedelta(weeks=int(candle * candle_multiplier))
                                        elif self.informative_timeframe.find('M') != -1:
                                            candle_time = \
                                                timeframe_to_prev_date(self.informative_timeframe, current_time) - \
                                                timedelta64(int(1 * candle_multiplier), 'M')

                                        candle_trend = \
                                            dataframe.loc[dataframe['date'] == candle_time].squeeze()['trend']

                                        if isinstance(candle_trend, str):
                                            stored_trend_dataframe[candle] = candle_trend
                                        else:
                                            break

                                    # Check if enough trend data has been stored to do the next check
                                    if len(stored_trend_dataframe) < \
                                            round(self.sell___unclogger_trend_lookback_candles_window.value /
                                                  self.precision):
                                        self.mgm_logger('debug', open_trade_unclogger,
                                                        f'No unclogging needed! Not enough trend data stored yet!')
                                    else:

                                        # Print all fetched 'trend' trade data
                                        self.mgm_logger('debug', open_trade_unclogger,
                                                        f'All needed "trend" trade data '
                                                        f'({str(len(stored_trend_dataframe))}) fetched!')
                                        self.mgm_logger('debug', open_trade_unclogger,
                                                        f'stored_trend_dataframe contents: '
                                                        f'{repr(stored_trend_dataframe)}')

                                        # Check if open_trade's trend changed negatively during past X candles
                                        self.mgm_logger('debug', open_trade_unclogger,
                                                        f'Calculating amount of unclogger_trend_lookback_candles_window'
                                                        f' "satisfied" for pair: {pair}')
                                        unclogger_candles_satisfied = 0
                                        for lookback_candle \
                                                in range(1,
                                                         round(self.sell___unclogger_trend_lookback_candles_window.value
                                                               / self.precision) + 1):
                                            if self.sell___unclogger_trend_lookback_window_uses_downwards_candles.value \
                                                    & (stored_trend_dataframe[lookback_candle] == 'downwards'):
                                                unclogger_candles_satisfied += 1
                                            if self.sell___unclogger_trend_lookback_window_uses_sideways_candles.value \
                                                    & (stored_trend_dataframe[lookback_candle] == 'sideways'):
                                                unclogger_candles_satisfied += 1
                                            if self.sell___unclogger_trend_lookback_window_uses_upwards_candles.value \
                                                    & (stored_trend_dataframe[lookback_candle] == 'upwards'):
                                                unclogger_candles_satisfied += 1
                                        self.mgm_logger('debug', open_trade_unclogger,
                                                        f'Amount of unclogger_trend_lookback_candles_window '
                                                        f'"satisfied": {str(unclogger_candles_satisfied)} '
                                                        f'for pair: {pair}')

                                        # Calculate the percentage of the lookback window currently satisfied
                                        unclogger_candles_percentage_satisfied = \
                                            (unclogger_candles_satisfied /
                                             round(self.sell___unclogger_trend_lookback_candles_window.value /
                                                   self.precision)) * 100

                                        # Override Sell Signal: Unclog trade by forcing a sell & attempt to continue
                                        # the profit climb with the "freed up trading slot"
                                        if unclogger_candles_percentage_satisfied >= \
                                                round(
                                                    self.sell___unclogger_trend_lookback_candles_window_percentage_needed.value
                                                    / self.precision):
                                            self.mgm_logger('info', open_trade_unclogger, f'Unclogging losing trade...')
                                            return "MGM_unclogging_losing_trade"
                                        else:
                                            self.mgm_logger('info', open_trade_unclogger,
                                                            f'No need to unclog open trade...')

            except Exception as e:
                self.mgm_logger('error', open_trade_unclogger,
                                f'Following error has occurred in the Open Trade Unclogger:')
                self.mgm_logger('error', open_trade_unclogger, str(e))

        return None  # By default we don't want a force sell to occur

    def mgm_logger(self, message_type: str, code_section: str, message: str):
        """
        MoniGoMani Logger:
        ---------------------
        When passing a type and a message to this function it will log:
        - The timestamp of logging + the message_type provided + the message provided
        - To the console & To "./user_data/logs/freqtrade.log"
    
        :param message_type: The type of the message (INFO, DEBUG, WARNING, ERROR)
        :param code_section: The section in the code where the message occurred
        :param message: The log message to be displayed
        """

        if self.use_mgm_logging:
            if (self.mgm_log_levels_enabled['info'] is True) and (message_type.upper() == 'INFO'):
                logger.setLevel(logging.INFO)
                logger.info(code_section + ' - ' + message)
            elif (self.mgm_log_levels_enabled['debug'] is True) and (message_type.upper() == 'DEBUG'):
                logger.setLevel(logging.DEBUG)
                logger.debug(code_section + ' - ' + message)
            elif (self.mgm_log_levels_enabled['warning'] is True) and (message_type.upper() == 'WARNING'):
                logger.setLevel(logging.WARNING)
                logger.warning(code_section + ' - ' + message)
            elif (self.mgm_log_levels_enabled['error'] is True) and (message_type.upper() == 'ERROR'):
                logger.setLevel(logging.ERROR)
                logger.error(code_section + ' - ' + message)
