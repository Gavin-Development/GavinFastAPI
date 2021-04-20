"""
 Copyright (c) 2020 Joshua Shiells

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import asyncio

from fastapi import FastAPI, Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from chat_bot import ChatBotTF

limiter = Limiter(key_func=get_remote_address)
api = FastAPI()
api.state.limiter = limiter
api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
fmt = '%d/%m/%Y %H-%M-%S.%f'
ChatBot = ChatBotTF.load("api_config.json")


@api.get(path='/')
@limiter.limit("10/second")
async def root(request: Request, response: Response):
    return {"paths": {route.path: route.name for route in api.routes[5:]}}


@api.get(path='/chat_bot/hparams')
@limiter.limit("40/second")
async def model_hparams(request: Request, response: Response):
    return ChatBot.hparams_dict


@api.get(path='/chat_bot/{msg}')
@limiter.limit("10/second")
async def chat_api(msg: str, request: Request, response: Response):
    try:
        bot_response = await asyncio.wait_for(ChatBot.predict_msg(msg), timeout=0.1)
        return {"message": bot_response}
    except TimeoutError:
        return {"error": "Message Timeout."}
