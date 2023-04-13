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

PARAM_FILE = "params.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "..", "data", "case2", "training_pricepaths.csv")

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
            await self.handle_market_snapshot(update.market_snapshot_msg)
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

    async def handle_market_snapshot(self, snapshot: pb.MarketSnapshotMessage):
        for asset_code, book in snapshot.books.items():
            option_type = asset_code[-1]  # Extract the option type (C or P)
            strike = int(asset_code[-4:-1])  # Extract the strike price

            # Find the row with the closest underlying price to the current price
            underlying_rows = self.option_data.iloc[(self.option_data['underlying'] - book.underlying_px).abs().argsort()[:1]]
            underlying_price = underlying_rows.iloc[0]

            if option_type == "C":
                option_price = underlying_price[f"call{strike}"]
            elif option_type == "P":
                option_price = underlying_price[f"put{strike}"]

            
            if book.bids and book.asks:
                bid_price = float(book.bids[0].px)
                ask_price = float(book.asks[0].px)
                
                if option_price < bid_price:
                    await self.place_order(asset_code, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.BUY, self.params["qty"], option_price)
                elif option_price > ask_price:
                    await self.place_order(asset_code, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.SELL, self.params["qty"], option_price)

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

if __name__ == "__main__":
    start_bot(BlackScholesBot)
