# Gavin FastAPI
This is part of the [Gavin](https://github.com/Scot-Survivor/GavinTraining) Family of Repos. A basic FastAPI
to be able to make http requests to Gavin. Everything is returned in JSON.

## Install
### Initial
#### Either
- `git submodule init` Initialise Submodule Git Repos.
- `git submodule update` Fetch all required repos.
#### Or
- `git clone https://github.com/Scot-Survivor/GavinFastAPI.git --recurse-submodules` To automatically 
run the above commands.
### Python  
- `python -m virtualenv ./GavinAPI/` Create Virtual Environment.
- Win: `./GavinAPI/Scripts/activate.bat` Linux/Mac: `source ./GavinAPI/bin/activate`
- `pip install -r requirements.txt` To install all required libraries inside venv. 

## Run
### With file coverage:
- `python3 main.py`
### Without file coverage:
Please note some settings api_config.json will not be parsed; you must manually edit the start.sh file:
`--host <str>` `--port <int>` `--workers <int>`
- `chmod +x start.sh`
- `./start.sh`


## Config
- `PREFORMER` (bool) True if the model is a preformer, false otherwise.
- `MODEL_DIR` (str) Path to where all the models will be stored such as "./models"
- `DEFAULT_MODEL_NAME` (str) Name of the model to load.
- `VERSION_OVERRIDE` (bool) Whether to override `DEFAULT_MODEL_NAME` and base it on versions instead.
- `VERSION` (str) If this is not blank, then load a model with this version.
- `HOST` (str) The host to bind to.
- `PORT` (int) The port to bind to.
- `FILTERED_WORDS` (list<str>) Words to censor on response from Gavin
- `TF_CPP_MIN_LOG_LEVEL` (str["0", "1", "2", "3"]) Tensorflow log level. 0 Most verbose, 3 Least verbose
- `UVICORN_WORKERS` (int) How many workers to use. 
- `MESSAGE_TIMEOUT` (int) How many seconds till the API returns "Message Timed Out Error."
- `CACHE_REQUEST_MAX` (int) How many requests should a Cache addition last for. 
- `MAX_CACHE_STORE` (int) Max amount of messages that should be stored in the cache.
- `LOGGING_LEVEL` (str["INFO", "WARN", "ERROR", "DEBUG"]) Log level of internal server.

# TODO
- Implement In Memory Cache (Added: 19/05/21)
- Implement Database (SQLite) based Cache (Added: 19/05/21)
- Implement Serving the associated Model Image (Added: 19/05/21)

## Licence
[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.txt). Should have a copy with this software.