import asyncio
import json
import logging
from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from GavinBackend.GavinCore.models import TransformerIntegration, PerformerIntegration
from utils import config_verification

limiter = Limiter(key_func=get_remote_address)
api = FastAPI()
api.state.limiter = limiter
api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
fmt = '%d/%m/%Y %H-%M-%S.%f'
api_config = json.load(open('api_config.json', 'rb'))
if api_config['PERFORMER']:
    ChatBot = PerformerIntegration.load_model(api_config["MODEL_DIR"], api_config["DEFAULT_MODEL_NAME"])
else:
    ChatBot = TransformerIntegration.load_model(api_config["MODEL_DIR"], api_config["DEFAULT_MODEL_NAME"])

is_valid, exception = config_verification(api_config)
if not is_valid:
    raise exception

MESSAGE_TIMEOUT = api_config['MESSAGE_TIMEOUT']
CACHE_REQUEST_MAX = api_config['CACHE_REQUEST_MAX']
MAX_CACHE_STORE = api_config['MAX_CACHE_STORE']
LOGGING_LEVELS = {'DEBUG': logging.DEBUG,
                  'INFO': logging.INFO,
                  'WARNING': logging.WARNING,
                  'ERROR': logging.ERROR,
                  'CRITICAL': logging.CRITICAL}

LOGGING_LEVEL = LOGGING_LEVELS[api_config['LOGGING_LEVEL']]
MESSAGE_CACHE = {}

logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def format_data(data: str):
    data = data.replace(' newlinechar ', '\n').replace(' newlinechar ', '\n').replace('"', "'")
    return data


class Message(BaseModel):
    """Message object for accepting json,
    this could be expanded to include more
    features, such as database storage"""
    data: str


@api.middleware('http')
async def msg_timeout(request: Request, call_next):
    """Times out a request after MESSAGE_TIMEOUT.
    Only on a call to TensorFlow."""
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
    """Returns a json of valid routes."""
    return {"paths": {route.path: route.name for route in api.routes[5:]}}


@api.get(path="/config")
@limiter.limit("1/second")
async def config(request: Request, response: Response):
    """Returns the config currently in process."""
    return api_config


@api.get(path='/chat_bot/hparams')
@limiter.limit("1/second")
async def model_hparams(request: Request, response: Response):
    """Return the Hyper parameters currently being used
    by the model"""
    hparams = ChatBot.get_hparams()
    hparams['TOKENIZER'] = f"Tokenizer Object. Vocab_Size: {ChatBot.vocab_size}"
    return hparams


@api.get(path='/chat_bot/metadata')
@limiter.limit("1/second")
async def model_metadata(request: Request, response: Response):
    """Return the Metadata that the current model version is using."""
    metadata = ChatBot.get_metadata()
    return metadata


@api.get(path='/chat_bot/model_name')
@limiter.limit("1/second")
async def model_name(request: Request, response: Response):
    """Return the model name."""
    return {"ModelName": ChatBot.name}


@api.post(path='/chat_bot/')
@limiter.limit("10/second")
async def chat_api(message: Message, request: Request, response: Response):
    """POST: Send a json object Message, get the bot_response from that."""
    if request.method == 'POST':
        if message.data in MESSAGE_CACHE.keys():
            bot_response_cache = MESSAGE_CACHE[message.data]
            bot_response = {"message": bot_response_cache[0]}
            if not bot_response_cache[1] + 1 > CACHE_REQUEST_MAX:
                MESSAGE_CACHE[message.data] = (bot_response_cache[0], bot_response_cache[1] + 1)
            else:
                del MESSAGE_CACHE[message.data]
            logging.debug(f"Cache Hit: {message.data}")
            return bot_response
        bot_response = format_data(ChatBot.predict(message.data))
        if not len(MESSAGE_CACHE.keys()) >= MAX_CACHE_STORE:
            MESSAGE_CACHE[message.data] = (bot_response, 0)
        logging.debug(f"Cache Miss: {message.data}")
        return {"message": bot_response}
