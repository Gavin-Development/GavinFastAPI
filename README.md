# Gavin FastAPI
This is part of the [Gavin](https://github.com/Scot-Survivor/GavinTraining) Family of Repos. A basic FastAPI
to be able to make http requests to Gavin. Everything is returned in JSON.

## Install
`pip install -r requirements.txt` To install all required libraries. 

## Run
Then run `python3 main.py` to start the server.

## Config
- `MODEL_DIR` (str) Path to where all the models are stored such as "./models"
- `DEFAULT_MODEL_NAME` (str) Name of the model to load.
- `VERSION_OVERRIDE` (bool) Whether to override `DEFAULT_MODEL_NAME` and base it on versions instead.
- `VERSION` (str) If this is not blank, then load a model with this version.
- `HOST` (str) The host to bind to.
- `PORT` (int) The port to bind to.
- `LOG_LEVEL` (str["INFO", "WARN", "ERROR"]) Log level of internal server (To come)
- `FILTERED_WORDS` (list<str>) Words to censor on response from Gavin
- `TF_CPP_MIN_LOG_LEVEL` (str["0", "1", "2", "3"]) Tensorflow log level. 0 Most verbose, 3 Least verbose
- `UVICORN_WORKERS` (int) How many workers to use. 
- `MESSAGE_TIMEOUT` (int) How many seconds till the API returns "Message Timed Out Error."

## Licence
[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.txt). Should have a copy with this software.