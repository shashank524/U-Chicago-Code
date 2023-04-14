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
import aiofiles


PARAM_FILE = "params.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "..", "data", "case2", "training_pricepaths.csv")

# Add a cache to store the calculated option prices
option_price_cache = {}

def round_to_tick_size(price, tick_size=0.1):
    return round(price / tick_size) * tick_size


class BlackScholesBot(UTCBot):
    def __init__(self, username, key, host, port):
        super().__init__(username, key, host, port)
        self.params = None
        self.option_data = pd.read_csv(CSV_FILE, index_col=0)
        self.last_underlying_price = 0
        self.price_update_threshold = 0.1
        self.params_ready = asyncio.Event()  # Add this line

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


    async def handle_read_params(self):
        while True:
            try:
                async with aiofiles.open(PARAM_FILE, "r") as f:
                    self.params = json.loads(await f.read())
                    self.params_ready.set()  # Add this line
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(1)





    async def handle_market_snapshot(self, snapshot: pb.MarketSnapshotMessage, underlying_price: float):
        await self.params_ready.wait()  # Add this line

        # Update the option prices in the cache only when the underlying price changes significantly
        if abs(self.last_underlying_price - underlying_price) >= self.price_update_threshold:
            self.last_underlying_price = underlying_price

            # Update the option prices in the cache
            for asset_code, book in snapshot.books.items():
                match = re.match(r"SPY(\d+(?:\.\d+)?)([CP])", asset_code)

                if match:
                    strike_price, option_type = match.groups()
                    strike_price = float(strike_price)

                    underlying_rows = self.option_data.iloc[(self.option_data['underlying'] - underlying_price).abs().argsort()[:1]]

                    underlying_price = underlying_rows.iloc[0]['underlying']  # Update this line

                    bs_params = self.params.copy()
                    bs_params["K"] = strike_price
                    option_price = black_scholes_binomial_opt(underlying_price, bs_params)

                    option_price_cache[asset_code] = option_price

        # Use the cached option prices for placing orders
        for asset_code, book in snapshot.books.items():
            option_price = option_price_cache.get(asset_code, None)

            if option_price is not None and book.bids and book.asks:
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

def black_scholes_binomial_opt(underlying_price, params, n_steps=252):
    S0 = underlying_price
    K = params["K"]
    T = params["T"]
    r = params["r"]
    sigma = params["sigma"]

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    call_payoffs = np.maximum(S0 * u**np.arange(n_steps + 1) * d**(n_steps - np.arange(n_steps + 1)) - K, 0)

    for i in range(n_steps - 1, -1, -1):
        call_payoffs[:-1] = np.exp(-r * dt) * (p * call_payoffs[:-1] + (1 - p) * call_payoffs[1:])

    call_price = call_payoffs[0]

    return call_price

if __name__ == "__main__":
    start_bot(BlackScholesBot)
