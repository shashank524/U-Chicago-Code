#!/usr/bin/env python

'''
A Black-Scholes model implementation
'''

import asyncio
import json
import pandas as pd
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
from math import sqrt, exp
from scipy.stats import norm
import os
import re
from math import log
import numpy as np
# from numba import jit


PARAM_FILE = "params.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "..", "data", "case2", "training_pricepaths.csv")

asset_code_pattern = re.compile(r"SPY(\d+(?:\.\d+)?)([CP])")


def round_to_tick_size(price, tick_size=0.1):
    return round(price / tick_size) * tick_size

class BlackScholesBot(UTCBot):
    def __init__(self, username, key, host, port):
        super().__init__(username, key, host, port)
        self.params = None
        self.option_data = pd.read_csv(CSV_FILE, index_col=0)
        self.asset_code_re = re.compile(r"SPY(\d+(?:\.\d+)?)([CP])")  # Precompile the regular expression
        self.closest_underlying_cache = {}  # Initialize the cache

    async def handle_round_started(self):
        await asyncio.sleep(0.1)
        asyncio.create_task(self.handle_read_params())

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")
        if kind == "market_snapshot_msg":
            # Obtain the underlying price from self.option_data
            underlying_price = self.option_data.loc[0, 'underlying']

            await self.handle_market_snapshot(update.market_snapshot_msg, underlying_price)
        elif kind == "generic_msg":
            msg = update.generic_msg.message
            print(msg)
            #!/usr/bin/env python

'''
Linear regression model to map weather data to futures prices
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from collections import defaultdict
from typing import DefaultDict, Dict, Tuple
from utc_bot import UTCBot, start_bot
import math
import proto.utc_bot as pb
import betterproto
import asyncio
import re
import os

DAYS_IN_MONTH = 21
DAYS_IN_YEAR = 252
INTEREST_RATE = 0.05
NUM_FUTURES = 14
TICK_SIZE = 0.00001
FUTURE_CODES = [chr(ord('A') + i) for i in range(NUM_FUTURES)] # Suffix of monthly future code
CONTRACTS = ['SBL'] +  ['LBS' + c for c in FUTURE_CODES] + ['LLL']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class Case1Bot(UTCBot):
    """
    An example bot
    """
    etf_suffix = ''
    async def create_etf(self, qty: int):
        '''
        Creates qty amount the ETF basket
        DO NOT CHANGE
        '''
        if len(self.etf_suffix) == 0:
            return pb.SwapResponse(False, "Unsure of swap")
        return await self.swap("create_etf_" + self.etf_suffix, qty)

    async def redeem_etf(self, qty: int):
        '''
        Redeems qty amount the ETF basket
        DO NOT CHANGE
        '''
        if len(self.etf_suffix) == 0:
            return pb.SwapResponse(False, "Unsure of swap")
        return await self.swap("redeem_etf_" + self.etf_suffix, qty)

    async def days_to_expiry(self, asset):
        '''
        Calculates days to expiry for the future
        '''
        future = ord(asset[-1]) - ord('A')
        expiry = 21 * (future + 1)
        return self._day - expiry

    async def handle_exchange_update(self, update: pb.FeedMessage):
        '''
        Handles exchange updates
        '''
        kind, _ = betterproto.which_one_of(update, "msg")
        #Competition event messages
        if kind == "generic_msg":
            msg = update.generic_msg.message

            # Used for API DO NOT TOUCH
            if 'trade_etf' in msg:
                self.etf_suffix = msg.split(' ')[1]

            # Updates current weather
            if "Weather" in update.generic_msg.message:
                msg = update.generic_msg.message
                weather = float(re.findall("\d+\.\d+", msg)[0])
                self._weather_log.append(weather)

            # Updates date
            if "Day" in update.generic_msg.message:
                self._day = int(re.findall("\d+", msg)[0])

            # Updates positions if unknown message (probably etf swap)
            else:
                resp = await self.get_positions()
                if resp.ok:
                    self.positions = resp.positions

        elif kind == "MarketSnapshotMessage":
            for asset in CONTRACTS:
                book = update.market_snapshot_msg.books[asset]
                self._best_bid[asset] = float(book.bids[0].px)
                self._best_ask[asset] = float(book.bids[0].px)


    async def handle_round_started(self):
        ### Current day
        self._day = 0
        ### Best Bid in the order book
        self._best_bid: Dict[str, float] = defaultdict(
            lambda: 0
        )
        ### Best Ask in the order book
        self._best_ask: Dict[str, float] = defaultdict(
            lambda: 0
        )
        ### Order book for market making
        self.__orders: DefaultDict[str, Tuple[str, float]] = defaultdict(
            lambda: ("", 0)
        )
        ### TODO Recording fair price for each asset
        self._fair_price: DefaultDict[str, float] = defaultdict(
            lambda: ("", 0)
        )
        ### TODO spread fair price for each asset
        self._spread: DefaultDict[str, float] = defaultdict(
            lambda: ("", 0)
        )

        ### TODO order size for market making positions
        self._quantity: DefaultDict[str, int] = defaultdict(
            lambda: ("", 0)
        )

        ### List of weather reports
        self._weather_log = []

        # Load CSV data
        weather_path = os.path.join(SCRIPT_DIR, "..", "data", "case1", "weather_2022.csv")
        weather_2020 = pd.read_csv(weather_path)
        # weather_2021 = pd.read_csv("weather_2021.csv")
        futures_path = os.path.join(SCRIPT_DIR, "..", "data", "case1", "futures_2022.csv")
        futures_2020 = pd.read_csv(futures_path)
        # futures_2021 = pd.read_csv("futures_2021_clean.csv")

        # Combine weather and futures data
        data_2022 = pd.concat([weather_2020, futures_2020], axis=1)
        # data_2022 = pd.concat([weather_2021, futures_2021], axis=1)
        combined_data = data_2022

        # Train linear regression model
        self.lr_model = LinearRegression()
        self.lr_model.fit(combined_data["weather"].values.reshape(-1, 1), combined_data["SBL"].values)

        await asyncio.sleep(.1)
        ###
        ### TODO START ASYNC FUNCTIONS HERE
        ###
        asyncio.create_task(self.example_redeem_etf())

        # Starts market making for each asset
        # for asset in CONTRACTS:
            # asyncio.create_task(self.make_market_asset(asset))

    # This is an example of creating and redeeming etfs
    # You can remove this in your actual bots.
    '''
    async def example_redeem_etf(self):
        while True:
            redeem_resp = await self.redeem_etf(1)
            create_resp = await self.create_etf(5)
            await asyncio.sleep(1)
    '''

    async def example_redeem_etf(self):
        while True:
            # Check positions to manage risk
            positions = await self.get_positions()
            if positions.ok:
                self.positions = positions.positions
                etf_position = self.positions.get("etf", 0)

                if etf_position <= 0:
                    # If ETF position is negative or zero, create ETFs
                    create_resp = await self.create_etf(1)
                else:
                    # If ETF position is positive, redeem ETFs
                    redeem_resp = await self.redeem_etf(1)

            await asyncio.sleep(1)




    ### Helpful ideas
    async def calculate_risk_exposure(self):
        pass

    async def calculate_fair_price(self, asset):
        if self._day < len(self._weather_log):
            predicted_weather = self.lr_model.predict(np.array(self._weather_log[self._day]).reshape(-1, 1))
            fair_price = max(0, predicted_weather[0])  # Ensure fair price is non-negative
            self._fair_price[asset] = fair_price
        else:
            self._fair_price[asset] = 0  # Set fair price to zero if no weather data is available

#!/usr/bin/env python

'''
Linear regression model to map weather data to futures prices
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from collections import defaultdict
from typing import DefaultDict, Dict, Tuple
from utc_bot import UTCBot, start_bot
import math
import proto.utc_bot as pb
import betterproto
import asyncio
import re
import os

DAYS_IN_MONTH = 21
DAYS_IN_YEAR = 252
INTEREST_RATE = 0.02
NUM_FUTURES = 14
TICK_SIZE = 0.00001
FUTURE_CODES = [chr(ord('A') + i) for i in range(NUM_FUTURES)] # Suffix of monthly future code
CONTRACTS = ['SBL'] +  ['LBS' + c for c in FUTURE_CODES] + ['LLL']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class Case1Bot(UTCBot):
    """
    An example bot
    """
    etf_suffix = ''
    async def create_etf(self, qty: int):
        '''
        Creates qty amount the ETF basket
        DO NOT CHANGE
        '''
        if len(self.etf_suffix) == 0:
            return pb.SwapResponse(False, "Unsure of swap")
        return await self.swap("create_etf_" + self.etf_suffix, qty)

    async def redeem_etf(self, qty: int):
        '''
        Redeems qty amount the ETF basket
        DO NOT CHANGE
        '''
        if len(self.etf_suffix) == 0:
            return pb.SwapResponse(False, "Unsure of swap")
        return await self.swap("redeem_etf_" + self.etf_suffix, qty)

    async def days_to_expiry(self, asset):
        '''
        Calculates days to expiry for the future
        '''
        future = ord(asset[-1]) - ord('A')
        expiry = 21 * (future + 1)
        return self._day - expiry

    async def handle_exchange_update(self, update: pb.FeedMessage):
        '''
        Handles exchange updates
        '''
        kind, _ = betterproto.which_one_of(update, "msg")
        #Competition event messages
        if kind == "generic_msg":
            msg = update.generic_msg.message

            # Used for API DO NOT TOUCH
            if 'trade_etf' in msg:
                self.etf_suffix = msg.split(' ')[1]

            # Updates current weather
            if "Weather" in update.generic_msg.message:
                msg = update.generic_msg.message
                weather = float(re.findall("\d+\.\d+", msg)[0])
                self._weather_log.append(weather)

            # Updates date
            if "Day" in update.generic_msg.message:
                self._day = int(re.findall("\d+", msg)[0])

            # Updates positions if unknown message (probably etf swap)
            else:
                resp = await self.get_positions()
                if resp.ok:
                    self.positions = resp.positions

            print("Current positions:")
            for asset, position in self.positions.items():
                print(f"{asset}: {position}")


        elif kind == "MarketSnapshotMessage":
            for asset in CONTRACTS:
                book = update.market_snapshot_msg.books[asset]
                self._best_bid[asset] = float(book.bids[0].px)
                self._best_ask[asset] = float(book.bids[0].px)


    async def handle_round_started(self):
        ### Current day
        self._day = 0
        ### Best Bid in the order book
        self._best_bid: Dict[str, float] = defaultdict(
            lambda: 0
        )
        ### Best Ask in the order book
        self._best_ask: Dict[str, float] = defaultdict(
            lambda: 0
        )
        ### Order book for market making
        self.__orders: DefaultDict[str, Tuple[str, float]] = defaultdict(
            lambda: ("", 0)
        )
        ### TODO Recording fair price for each asset
        self._fair_price: DefaultDict[str, float] = defaultdict(
            lambda: ("", 0)
        )
        ### TODO spread fair price for each asset
        self._spread: DefaultDict[str, float] = defaultdict(
            lambda: ("", 0)
        )

        ### TODO order size for market making positions
        self._quantity: DefaultDict[str, int] = defaultdict(
            lambda: ("", 0)
        )

        ### List of weather reports
        self._weather_log = []

        # Load CSV data
        weather_path = os.path.join(SCRIPT_DIR, "..", "data", "case1", "weather_2022.csv")
        weather_2020 = pd.read_csv(weather_path)
        # weather_2021 = pd.read_csv("weather_2021.csv")
        futures_path = os.path.join(SCRIPT_DIR, "..", "data", "case1", "futures_2022.csv")
        futures_2020 = pd.read_csv(futures_path)
        # futures_2021 = pd.read_csv("futures_2021_clean.csv")

        # Combine weather and futures data
        data_2022 = pd.concat([weather_2020, futures_2020], axis=1)
        # data_2022 = pd.concat([weather_2021, futures_2021], axis=1)
        combined_data = data_2022

        # Train linear regression model
        self.lr_model = LinearRegression()
        self.lr_model.fit(combined_data["weather"].values.reshape(-1, 1), combined_data["SBL"].values)

        await asyncio.sleep(.1)
        ###
        ### TODO START ASYNC FUNCTIONS HERE
        ###
        asyncio.create_task(self.example_redeem_etf())

        # Starts market making for each asset
        # for asset in CONTRACTS:
            # asyncio.create_task(self.make_market_asset(asset))

    # This is an example of creating and redeeming etfs
    # You can remove this in your actual bots.
    '''
    async def example_redeem_etf(self):
        while True:
            redeem_resp = await self.redeem_etf(1)
            create_resp = await self.create_etf(5)
            await asyncio.sleep(1)
    '''

    async def example_redeem_etf(self):
        while True:
            # Check positions to manage risk
            positions = await self.get_positions()
            if positions.ok:
                self.positions = positions.positions
                etf_position = self.positions.get("etf", 0)

                if etf_position <= 0:
                    # If ETF position is negative or zero, create ETFs
                    create_resp = await self.create_etf(1)
                else:
                    # If ETF position is positive, redeem ETFs
                    redeem_resp = await self.redeem_etf(1)

            await asyncio.sleep(1)




    ### Helpful ideas
    async def calculate_risk_exposure(self):
        pass

    async def calculate_fair_price(self, asset):
        if self._day < len(self._weather_log):
            predicted_weather = self.lr_model.predict(np.array(self._weather_log[self._day]).reshape(-1, 1))
            fair_price = max(0, predicted_weather[0])  # Ensure fair price is non-negative
            self._fair_price[asset] = fair_price
        else:
            self._fair_price[asset] = 0  # Set fair price to zero if no weather data is available

    async def make_market_asset(self, asset: str):
        while self._day <= DAYS_IN_YEAR:
            ## Old prices
            ub_oid, ub_price = self.__orders["underlying_bid_{}".format(asset)]
            ua_oid, ua_price = self.__orders["underlying_ask_{}".format(asset)]

            # Calculate new bid and ask prices based on the updated fair price and spread
            bid_px = self._fair_price[asset] - self._spread[asset]
            ask_px = self._fair_price[asset] + self._spread[asset]

            # If the bid or ask prices have changed, cancel the old orders and place new ones
            if self._best_bid[asset] != bid_px or self._best_ask[asset] != ask_px:

                # Cancel old bid order
                if ub_oid:
                    await self.cancel_order(ub_oid)
                    self.__orders["underlying_bid_{}".format(asset)] = ("", 0)

                # Cancel old ask order
                if ua_oid:
                    await self.cancel_order(ua_oid)
                    self.__orders["underlying_ask_{}".format(asset)] = ("", 0)

                # Place new bid order
                ub_resp = await self.place_order(asset, True, bid_px, self._quantity[asset])
                if ub_resp.ok:
                    self.__orders["underlying_bid_{}".format(asset)] = (ub_resp.order_id, ub_resp.order.price)

                # Place new ask order
                ua_resp = await self.place_order(asset, False, ask_px, self._quantity[asset])
                if ua_resp.ok:
                    self.__orders["underlying_ask_{}".format(asset)] = (ua_resp.order_id, ua_resp.order.price)

            await asyncio.sleep(0.1)




def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))



if __name__ == "__main__":
    start_bot(Case1Bot)






def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))



if __name__ == "__main__":
    start_bot(Case1Bot)





    async def handle_read_params(self):
        while True:
            try:
                self.params = json.load(open(PARAM_FILE, "r"))
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(1)


    async def handle_market_snapshot(self, snapshot: pb.MarketSnapshotMessage, underlying_price: float):
        if underlying_price not in self.closest_underlying_cache:
            # Get the closest underlying price row in the option data and store it in the cache
            underlying_rows = self.option_data.iloc[(self.option_data['underlying'] - underlying_price).abs().argsort()[:1]]
            self.closest_underlying_cache[underlying_price] = underlying_rows.iloc[0]
            
        closest_underlying = self.closest_underlying_cache[underlying_price]

        for asset_code, book in snapshot.books.items():
            # Extract the strike price and option type from the asset_code using the precompiled regular expression
            match = self.asset_code_re.match(asset_code)

            if match:
                strike_price, option_type = match.groups()
                strike_price = float(strike_price)

                # Calculate the option price using the Black-Scholes model
                bs_params = self.params.copy()
                bs_params["K"] = strike_price
                option_price = black_scholes_binomial(closest_underlying, bs_params)


                if option_type == "C":
                    # option_price = underlying_price[f"call{strike_price}"]
                    option_price = underlying_price[f"call{int(strike_price)}"]
                elif option_type == "P":
                    option_price = underlying_price[f"put{int(strike_price)}"]
                    # option_price = underlying_price[f"put{strike_price}"]

                if book.bids and book.asks:
                    bid_price = float(book.bids[0].px)
                    ask_price = float(book.asks[0].px)

                    rounded_option_price = round_to_tick_size(option_price)

                    if option_price < bid_price:
                        print("placing order")
                        await self.place_order(asset_code, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.BID, self.params["qty"], rounded_option_price)
                        print("placed order")
                    elif option_price > ask_price:
                        print("placing order")
                        await self.place_order(asset_code, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.ASK, self.params["qty"], rounded_option_price)
                        print("placed order")



    def black_scholes(self, underlying_price, params):
        S = underlying_price["underlying"]
        K = params["K"]
        T = params["T"]
        r = params["r"]
        sigma = params["sigma"]

        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        call_price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        return call_price
    
# @jit(nopython=True)
def black_scholes_binomial(underlying_price, params, n_steps=252):
    S0 = underlying_price["underlying"]
    K = params["K"]
    T = params["T"]
    r = params["r"]
    sigma = params["sigma"]

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Vectorized binomial tree calculation
    # tree = np.zeros((n_steps + 1, n_steps + 1))
    tree = np.empty((n_steps + 1, n_steps + 1))
    j = np.arange(n_steps + 1)
    i = np.arange(n_steps + 1)[:, np.newaxis]
    tree[i, j] = S0 * (u ** (i - 2 * j)) * (i >= j)

    call_payoffs = np.maximum(tree[-1] - K, 0)
    q = np.exp(-r * dt)

    for i in range(n_steps - 1, -1, -1):
        call_payoffs[:i+1] = q * (p * call_payoffs[:i+1] + (1 - p) * call_payoffs[1:i+2])

    call_price = call_payoffs[0]

    return call_price


if __name__ == "__main__":
    start_bot(BlackScholesBot)
