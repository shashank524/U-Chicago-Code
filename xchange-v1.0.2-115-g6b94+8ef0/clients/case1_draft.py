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

