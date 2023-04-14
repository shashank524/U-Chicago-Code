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


PARAM_FILE = "params.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "..", "data", "case2", "training_pricepaths.csv")


def round_to_tick_size(price, tick_size=0.1):
    return round(price / tick_size) * tick_size

class BlackScholesBot(UTCBot):
    def __init__(self, username, key, host, port):
        super().__init__(username, key, host, port)
        self.params = None
        self.option_data = pd.read_csv(CSV_FILE, index_col=0)



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
                self.params = json.load(open(PARAM_FILE, "r"))
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(1)


    async def handle_market_snapshot(self, snapshot: pb.MarketSnapshotMessage, underlying_price: float):
        for asset_code, book in snapshot.books.items():
            # Extract the strike price and option type from the asset_code
            match = re.match(r"SPY(\d+(?:\.\d+)?)([CP])", asset_code)

            if match:
                strike_price, option_type = match.groups()
                strike_price = float(strike_price)

                # Get the closest underlying price row in the option data
                underlying_rows = self.option_data.iloc[(self.option_data['underlying'] - underlying_price).abs().argsort()[:1]]

                underlying_price = underlying_rows.iloc[0]

                # Calculate the option price using the Black-Scholes model
                bs_params = self.params.copy()
                bs_params["K"] = strike_price
                option_price = self.black_scholes_binomial_opt(underlying_price, bs_params)

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
    
    import numpy as np

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

    S = np.zeros((n_steps + 1, n_steps + 1))
    S[0, 0] = S0

    for i in range(1, n_steps + 1):
        S[i, 0] = S[i-1, 0] * u
        for j in range(1, i + 1):
            S[i, j] = S[i-1, j-1] * d

    call_payoffs = np.maximum(S[-1] - K, 0)

    for i in range(n_steps - 1, -1, -1):
        call_payoffs = np.exp(-r * dt) * (p * call_payoffs[:-1] + (1 - p) * call_payoffs[1:])

    call_price = call_payoffs[0]

    return call_price

def black_scholes_binomial_opt(underlying_price, params, n_steps=252):
    S0 = underlying_price["underlying"]
    K = params["K"]
    T = params["T"]
    r = params["r"]
    sigma = params["sigma"]

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    call_payoffs = np.zeros(n_steps + 1)
    for j in range(n_steps + 1):
        call_payoffs[j] = max(S0 * u**j * d**(n_steps - j) - K, 0)

    for i in range(n_steps - 1, -1, -1):
        call_payoffs[:-1] = np.exp(-r * dt) * (p * call_payoffs[:-1] + (1 - p) * call_payoffs[1:])

    call_price = call_payoffs[0]

    return call_price




if __name__ == "__main__":
    start_bot(BlackScholesBot)
