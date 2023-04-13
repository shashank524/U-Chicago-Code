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

PARAM_FILE = "params.json"
CSV_FILE = "training_pricepaths.csv"

class BlackScholesBot(UTCBot):
    def __init__(self):
        super().__init__()
        self.params = None
        self.option_data = pd.read_csv(CSV_FILE)

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
            underlying_price = self.option_data.loc[self.option_data['underlying'] == asset_code].iloc[0]
            option_price = self.black_scholes(underlying_price, self.params)
            
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
