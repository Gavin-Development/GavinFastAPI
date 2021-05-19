import asyncio
import json
from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from GavinBackend.GavinCore.models import TransformerIntegration

limiter = Limiter(key_func=get_remote_address)
api = FastAPI()
api.state.limiter = limiter
api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
fmt = '%d/%m/%Y %H-%M-%S.%f'
api_config = json.load(open('api_config.json', 'rb'))
ChatBot = TransformerIntegration.load_model(api_config["MODEL_DIR"], api_config["DEFAULT_MODEL_NAME"])
MESSAGE_TIMEOUT = api_config['MESSAGE_TIMEOUT']
CACHE_REQUEST_MAX = api_config['CACHE_REQUEST_MAX']
MESSAGE_CACHE = {}


class Message(BaseModel):
    """Message object for accepting json,
    this could be expanded to inculde more
    features, such as database storage"""
    data: str


@api.middleware('http')
async def msg_timeout(request: Request, call_next):
    if (request.url.path == '/chat_bot' or request.url.path == "/chat_bot/") and request.method == 'POST':
        try:
            response = await asyncio.wait_for(call_next(request), timeout=MESSAGE_TIMEOUT)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail=f"Message Timed Out. Took longer than: {MESSAGE_TIMEOUT}")
        else:
            return response
    else:
        response = await call_next(request)
        return response


@api.get(path='/')
@limiter.limit("1/second")
async def root(request: Request, response: Response):
    return {"paths": {route.path: route.name for route in api.routes[5:]}}


@api.get(path="/config")
@limiter.limit("1/second")
async def config(request: Request, response: Response):
    return api_config


@api.get(path='/chat_bot/hparams')
@limiter.limit("1/second")
async def model_hparams(request: Request, response: Response):
    hparams = ChatBot.get_hparams()
    hparams['TOKENIZER'] = f"Tokenizer Object. Vocab_Size: {ChatBot.vocab_size}"
    return hparams


@api.get(path='/chat_bot/model_name')
@limiter.limit("1/second")
async def model_name(request: Request, response: Response):
    return {"ModelName": ChatBot.name}


@api.post(path='/chat_bot/')
@limiter.limit("10/second")
async def chat_api(message: Message, request: Request, response: Response):
    if request.method == 'POST':
        if message.data in MESSAGE_CACHE.keys():
            bot_response_cache = MESSAGE_CACHE[message.data]
            bot_response = {"message": bot_response_cache[0]}
            if not bot_response_cache[1]+1 > CACHE_REQUEST_MAX:
                MESSAGE_CACHE[message.data] = (bot_response_cache[0], bot_response_cache[1]+1)
            else:
                del MESSAGE_CACHE[message.data]
            return bot_response
        bot_response = ChatBot.predict(message.data)
        MESSAGE_CACHE[message.data] = (bot_response, 0)
        return {"message": bot_response}
