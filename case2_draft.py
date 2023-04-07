#!/usr/bin/env python

'''
A Black-Scholes model implementation
'''

from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import asyncio
import json
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Import additional libraries
import os

PARAM_FILE = "params.json"
CSV_FILE = "training_pricepaths.csv"

# Read the CSV file into a DataFrame
price_data = pd.read_csv(CSV_FILE)

# Black-Scholes formula
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(S, K, T, r, C):
    def objective_function(sigma):
        return black_scholes_call(S, K, T, r, sigma) - C

    return brentq(objective_function, 1e-6, 1e6)

class OptionBot(UTCBot):
    """
    An example bot that reads from a file to set internal parameters during the round
    """

    async def handle_round_started(self):
        await asyncio.sleep(0.1)
        asyncio.create_task(self.handle_read_params())

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")
        # Competition event messages
        if kind == "generic_msg":
            msg = update.generic_msg.message
            print(msg)

        # Add your Black-Scholes model logic here

    async def handle_read_params(self):
        while True:
            try:
                self.params = json.load(open(PARAM_FILE, "r"))
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(1)

if __name__ == "__main__":
    start_bot(OptionBot)

