import h5py 
import numpy as np
from typing import * 
import tensorflow as tf
import matplotlib.pyplot as plt 
import random
import os 
from telegram import Bot
import glob
import keras
import time 
import sys
import asyncio

async def send_sms_to_me(sms: str) -> None:
    '''
    Description
        This function makes use of the Amaro-TelegramBot to send a message when the code is finished. 

            await send_sms_to_me('Hi')

    Args
        sms (str): Refers to the message to be send 
   '''
    BOT_TOKEN =
    CHAT_ID = 
    bot = Bot(token = BOT_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text = sms)
