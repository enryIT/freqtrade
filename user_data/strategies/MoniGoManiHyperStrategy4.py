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


class MoniGoManiHyperStrategy4(IStrategy):
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
        "buy__downwards_trend_total_signal_needed": 165,
        "buy__downwards_trend_total_signal_needed_candles_lookback_window": 2,
        "buy__sideways_trend_total_signal_needed": 870,
        "buy__sideways_trend_total_signal_needed_candles_lookback_window": 1,
        "buy__upwards_trend_total_signal_needed": 306,
        "buy__upwards_trend_total_signal_needed_candles_lookback_window": 4,
        "buy_downwards_trend_adx_strong_up_weight": 88,
        "buy_downwards_trend_bollinger_bands_weight": 91,
        "buy_downwards_trend_ema_long_golden_cross_weight": 72,
        "buy_downwards_trend_ema_short_golden_cross_weight": 38,
        "buy_downwards_trend_macd_weight": 96,
        "buy_downwards_trend_rsi_weight": 94,
        "buy_downwards_trend_sma_long_golden_cross_weight": 23,
        "buy_downwards_trend_sma_short_golden_cross_weight": 92,
        "buy_downwards_trend_vwap_cross_weight": 18,
        "buy_sideways_trend_adx_strong_up_weight": 60,
        "buy_sideways_trend_bollinger_bands_weight": 63,
        "buy_sideways_trend_ema_long_golden_cross_weight": 53,
        "buy_sideways_trend_macd_weight": 65,
        "buy_sideways_trend_sma_long_golden_cross_weight": 53,
        "buy_sideways_trend_vwap_cross_weight": 72,
        "buy_upwards_trend_adx_strong_up_weight": 85,
        "buy_upwards_trend_bollinger_bands_weight": 42,
        "buy_upwards_trend_ema_long_golden_cross_weight": 100,
        "buy_upwards_trend_ema_short_golden_cross_weight": 21,
        "buy_upwards_trend_macd_weight": 70,
        "buy_upwards_trend_rsi_weight": 31,
        "buy_upwards_trend_sma_long_golden_cross_weight": 9,
        "buy_upwards_trend_sma_short_golden_cross_weight": 50,
        "buy_upwards_trend_vwap_cross_weight": 28,
        "buy___trades_when_downwards": True,  # value loaded from strategy
        "buy___trades_when_sideways": False,  # value loaded from strategy
        "buy___trades_when_upwards": True,  # value loaded from strategy
        "buy_sideways_trend_ema_short_golden_cross_weight": 100,  # value loaded from strategy
        "buy_sideways_trend_rsi_weight": 0,  # value loaded from strategy
        "buy_sideways_trend_sma_short_golden_cross_weight": 100,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "sell___unclogger_minimal_losing_trade_duration_minutes": 27,
        "sell___unclogger_minimal_losing_trades_open": 5,
        "sell___unclogger_open_trades_losing_percentage_needed": 60,
        "sell___unclogger_trend_lookback_candles_window": 57,
        "sell___unclogger_trend_lookback_candles_window_percentage_needed": 37,
        "sell__downwards_trend_total_signal_needed": 212,
        "sell__downwards_trend_total_signal_needed_candles_lookback_window": 1,
        "sell__sideways_trend_total_signal_needed": 151,
        "sell__sideways_trend_total_signal_needed_candles_lookback_window": 3,
        "sell__upwards_trend_total_signal_needed": 767,
        "sell__upwards_trend_total_signal_needed_candles_lookback_window": 4,
        "sell_downwards_trend_adx_strong_down_weight": 79,
        "sell_downwards_trend_bollinger_bands_weight": 40,
        "sell_downwards_trend_ema_long_death_cross_weight": 17,
        "sell_downwards_trend_ema_short_death_cross_weight": 75,
        "sell_downwards_trend_macd_weight": 15,
        "sell_downwards_trend_rsi_weight": 51,
        "sell_downwards_trend_sma_long_death_cross_weight": 81,
        "sell_downwards_trend_sma_short_death_cross_weight": 81,
        "sell_downwards_trend_vwap_cross_weight": 60,
        "sell_sideways_trend_adx_strong_down_weight": 43,
        "sell_sideways_trend_bollinger_bands_weight": 88,
        "sell_sideways_trend_ema_long_death_cross_weight": 97,
        "sell_sideways_trend_ema_short_death_cross_weight": 56,
        "sell_sideways_trend_macd_weight": 47,
        "sell_sideways_trend_rsi_weight": 66,
        "sell_sideways_trend_sma_long_death_cross_weight": 12,
        "sell_sideways_trend_sma_short_death_cross_weight": 43,
        "sell_sideways_trend_vwap_cross_weight": 6,
        "sell_upwards_trend_adx_strong_down_weight": 4,
        "sell_upwards_trend_bollinger_bands_weight": 8,
        "sell_upwards_trend_ema_long_death_cross_weight": 58,
        "sell_upwards_trend_ema_short_death_cross_weight": 62,
        "sell_upwards_trend_macd_weight": 41,
        "sell_upwards_trend_rsi_weight": 68,
        "sell_upwards_trend_sma_long_death_cross_weight": 96,
        "sell_upwards_trend_sma_short_death_cross_weight": 12,
        "sell_upwards_trend_vwap_cross_weight": 4,
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
        "0": 0.3,
        "5": 0.29854,
        "10": 0.29708,
        "15": 0.29563,
        "20": 0.29417,
        "25": 0.29271,
        "30": 0.29125,
        "35": 0.28979,
        "40": 0.28834,
        "45": 0.28688,
        "50": 0.28542,
        "55": 0.28396,
        "60": 0.28251,
        "65": 0.28105,
        "70": 0.27959,
        "75": 0.27813,
        "80": 0.27667,
        "85": 0.27522,
        "90": 0.27376,
        "95": 0.2723,
        "100": 0.27084,
        "105": 0.26938,
        "110": 0.26793,
        "115": 0.26647,
        "120": 0.26501,
        "125": 0.26355,
        "130": 0.2621,
        "135": 0.26064,
        "140": 0.25918,
        "145": 0.25772,
        "150": 0.25626,
        "155": 0.25481,
        "160": 0.25335,
        "165": 0.25189,
        "170": 0.25043,
        "175": 0.24897,
        "180": 0.24752,
        "185": 0.24606,
        "190": 0.2446,
        "195": 0.24314,
        "200": 0.24169,
        "205": 0.24023,
        "210": 0.23877,
        "215": 0.23731,
        "220": 0.23585,
        "225": 0.2344,
        "230": 0.23294,
        "235": 0.23148,
        "240": 0.23002,
        "245": 0.22856,
        "250": 0.22711,
        "255": 0.22565,
        "260": 0.22419,
        "265": 0.22273,
        "270": 0.22128,
        "275": 0.21982,
        "280": 0.21836,
        "285": 0.2169,
        "290": 0.21544,
        "295": 0.21399,
        "300": 0.21253,
        "305": 0.21107,
        "310": 0.20961,
        "315": 0.20815,
        "320": 0.2067,
        "325": 0.20524,
        "330": 0.20378,
        "335": 0.20232,
        "340": 0.20087,
        "345": 0.19941,
        "350": 0.19795,
        "355": 0.19649,
        "360": 0.19503,
        "365": 0.19358,
        "370": 0.19212,
        "375": 0.19066,
        "380": 0.1892,
        "385": 0.18774,
        "390": 0.18629,
        "395": 0.18483,
        "400": 0.18337,
        "405": 0.18191,
        "410": 0.18046,
        "415": 0.179,
        "420": 0.17754,
        "425": 0.17608,
        "430": 0.17462,
        "435": 0.17317,
        "440": 0.17175,
        "445": 0.17047,
        "450": 0.1692,
        "455": 0.16792,
        "460": 0.16665,
        "465": 0.16538,
        "470": 0.1641,
        "475": 0.16283,
        "480": 0.16156,
        "485": 0.16028,
        "490": 0.15901,
        "495": 0.15773,
        "500": 0.15646,
        "505": 0.15519,
        "510": 0.15391,
        "515": 0.15264,
        "520": 0.15137,
        "525": 0.15009,
        "530": 0.14882,
        "535": 0.14755,
        "540": 0.14627,
        "545": 0.145,
        "550": 0.14372,
        "555": 0.14245,
        "560": 0.14118,
        "565": 0.1399,
        "570": 0.13863,
        "575": 0.13736,
        "580": 0.13608,
        "585": 0.13481,
        "590": 0.13353,
        "595": 0.13226,
        "600": 0.13099,
        "605": 0.12971,
        "610": 0.12844,
        "615": 0.12717,
        "620": 0.12589,
        "625": 0.12462,
        "630": 0.12335,
        "635": 0.12207,
        "640": 0.1208,
        "645": 0.11952,
        "650": 0.11825,
        "655": 0.11698,
        "660": 0.1157,
        "665": 0.11443,
        "670": 0.11316,
        "675": 0.11188,
        "680": 0.11061,
        "685": 0.10933,
        "690": 0.10806,
        "695": 0.10679,
        "700": 0.10551,
        "705": 0.10424,
        "710": 0.10297,
        "715": 0.10169,
        "720": 0.10042,
        "725": 0.09915,
        "730": 0.09787,
        "735": 0.0966,
        "740": 0.09532,
        "745": 0.09405,
        "750": 0.09278,
        "755": 0.0915,
        "760": 0.09023,
        "765": 0.08896,
        "770": 0.08768,
        "775": 0.08641,
        "780": 0.08513,
        "785": 0.08386,
        "790": 0.08259,
        "795": 0.08131,
        "800": 0.08004,
        "805": 0.07877,
        "810": 0.07749,
        "815": 0.07622,
        "820": 0.07495,
        "825": 0.07367,
        "830": 0.0724,
        "835": 0.07112,
        "840": 0.06985,
        "845": 0.06858,
        "850": 0.0673,
        "855": 0.06603,
        "860": 0.06476,
        "865": 0.06348,
        "870": 0.06221,
        "875": 0.06093,
        "880": 0.05966,
        "885": 0.05839,
        "890": 0.05711,
        "895": 0.05584,
        "900": 0.05457,
        "905": 0.05329,
        "910": 0.05202,
        "915": 0.0508,
        "920": 0.0498,
        "925": 0.04879,
        "930": 0.04779,
        "935": 0.04678,
        "940": 0.04578,
        "945": 0.04478,
        "950": 0.04377,
        "955": 0.04277,
        "960": 0.04176,
        "965": 0.04076,
        "970": 0.03976,
        "975": 0.03875,
        "980": 0.03775,
        "985": 0.03674,
        "990": 0.03574,
        "995": 0.03474,
        "1000": 0.03373,
        "1005": 0.03273,
        "1010": 0.03172,
        "1015": 0.03072,
        "1020": 0.02972,
        "1025": 0.02871,
        "1030": 0.02771,
        "1035": 0.0267,
        "1040": 0.0257,
        "1045": 0.0247,
        "1050": 0.02369,
        "1055": 0.02269,
        "1060": 0.02169,
        "1065": 0.02068,
        "1070": 0.01968,
        "1075": 0.01867,
        "1080": 0.01767,
        "1085": 0.01667,
        "1090": 0.01566,
        "1095": 0.01466,
        "1100": 0.01365,
        "1105": 0.01265,
        "1110": 0.01165,
        "1115": 0.01064,
        "1120": 0.00964,
        "1125": 0.00863,
        "1130": 0.00763,
        "1135": 0.00663,
        "1140": 0.00562,
        "1145": 0.00462,
        "1150": 0.00361,
        "1155": 0.00261,
        "1160": 0.00161,
        "1165": 0.0006,
        "1170": 0
    }

    # Stoploss:
    stoploss = -0.102

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.012
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
