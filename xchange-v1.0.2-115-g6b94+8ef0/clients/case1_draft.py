#!/usr/bin/env python

from collections import defaultdict
from typing import DefaultDict, Dict, Tuple
from utc_bot import UTCBot, start_bot
import math
import proto.utc_bot as pb
import betterproto
import asyncio
import re

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

weather_2020_file = 'weather_2020.csv'
weather_2021_file = 'weather_2022.csv'
futures_2020_file = 'futures_2020_clean.csv'
futures_2021_file = 'futures_2022.csv'

# weather_2020_file = os.path.join(SCRIPT_DIR, "..", "data", "case1", weather_2020_file)
weather_2021_file = os.path.join(SCRIPT_DIR, "..", "data", "case1", weather_2021_file)
# futures_2020_file = os.path.join(SCRIPT_DIR, "..", "data", "case2", futures_2020_file)
futures_2021_file = os.path.join(SCRIPT_DIR, "..", "data", "case1", futures_2021_file)

def read_weather_data(file_path: str) -> Dict[int, float]:
    weather_data = {}

    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip the header
        for row in reader:
            day, weather = int(row[0]), float(row[1])
            weather_data[day] = weather

    return weather_data

# weather_2020_data = read_weather_data(weather_2020_file)
weather_2021_data = read_weather_data(weather_2021_file)


def read_futures_data(file_path: str) -> Dict[int, Dict[str, float]]:
    futures_data = {}

    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # Get the header
        for row in reader:
            day = int(row[0])
            futures_data[day] = {header[i]: float(value) for i, value in enumerate(row[1:], start=1)}

    return futures_data

# futures_2020_data = read_futures_data(futures_2020_file)
futures_2021_data = read_futures_data(futures_2021_file)

def weather_impact(weather_data: Dict[int, float], day: int) -> float:
    # You can implement your own logic to calculate the weather impact on the fair price
    # For example, you can calculate the average weather value over a certain period
    weather_avg = sum(weather_data[day - i] for i in range(5)) / 5
    impact = (weather_data[day] - weather_avg) * 0.01  # Assuming 1% impact for each unit of weather difference
    return impact



DAYS_IN_MONTH = 21
DAYS_IN_YEAR = 252
INTEREST_RATE = 0.02
NUM_FUTURES = 14
TICK_SIZE = 0.00001
FUTURE_CODES = [chr(ord('A') + i) for i in range(NUM_FUTURES)] # Suffix of monthly future code
CONTRACTS = ['SBL'] +  ['LBS' + c for c in FUTURE_CODES] + ['LLL']
POSITION_LIMIT = 1000


class Case1Bot(UTCBot):
    """
    An example bot
    """
    async def is_within_position_limit(self, asset: str, new_position: int) -> bool:
        position = self.positions.get(asset, 0)
        return abs(position + new_position) <= POSITION_LIMIT

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
    async def example_redeem_etf(self):
        while True:
            create_resp = await self.create_etf(1)
            await asyncio.sleep(1)
            redeem_resp = await self.redeem_etf(1)
            await asyncio.sleep(1)



    ### Helpful ideas
    async def calculate_risk_exposure(self):
        pass
    
    async def calculate_fair_price(self, asset: str, day: int) -> float:
        # Example: Calculate fair price based on the mid-price of the best bid and ask prices
        mid_price = (self._best_bid[asset] + self._best_ask[asset]) / 2

        # Calculate the weather impact
        weather_impact_value = weather_impact(weather_2021_data, day)  # or use weather_2021_data for 2021

        # Adjust the fair price based on the weather impact
        adjusted_fair_price = mid_price + weather_impact_value

        # Incorporate futures data into the fair price calculation
        if asset.startswith('LBS'):
            future_data = futures_2021_data.get(day)
            if future_data:
                future_price = future_data.get(asset)
                if future_price:
                    adjusted_fair_price = (adjusted_fair_price + future_price) / 2

        return adjusted_fair_price



    async def calculate_spread(self, asset: str) -> float:
        # Example: Calculate spread as a percentage of the fair price
        spread_percentage = 0.01  # This could be adjusted based on your strategy
        spread = await self.calculate_fair_price(asset) * spread_percentage
        return spread
        
                
    async def make_market_asset(self, asset: str):
        while self._day <= DAYS_IN_YEAR:
            # Calculate fair price and spread dynamically
            fair_price = await self.calculate_fair_price(asset, self._day)
            spread = await self.calculate_spread(asset)

            # Old prices
            ub_oid, ub_price = self.__orders["underlying_bid_{}".format(asset)]
            ua_oid, ua_price = self.__orders["underlying_ask_{}".format(asset)]

            bid_px = fair_price - spread
            ask_px = fair_price + spread

            # If the underlying price moved first, adjust the ask first to avoid self-trades
            if (bid_px + ask_px) > (ua_price + ub_price):
                order = ["ask", "bid"]
            else:
                order = ["bid", "ask"]

            for d in order:
                if d == "bid":
                    order_id = ub_oid
                    order_side = pb.OrderSpecSide.BID
                    order_px = bid_px
                    new_position = -self._quantity[asset]  # Position change for a bid order
                else:
                    order_id = ua_oid
                    order_side = pb.OrderSpecSide.ASK
                    order_px = ask_px
                    new_position = self._quantity[asset]   # Position change for an ask order

                # Check if the new position is within the allowed position limit
                if await self.is_within_position_limit(asset, new_position):
                    r = await self.modify_order(
                            order_id=order_id,
                            asset_code=asset,
                            order_type=pb.OrderSpecType.LIMIT,
                            order_side=order_side,
                            qty=self._quantity[asset],
                            px=round_nearest(order_px, TICK_SIZE),
                        )

                    self.__orders[f"underlying_{d}_{asset}"] = (r.order_id, order_px)


        

def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))             



if __name__ == "__main__":
    start_bot(Case1Bot)

