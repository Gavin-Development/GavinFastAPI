from typing import Union, Tuple

from exceptions import InvalidConfig


def config_verification(config: dict) -> Union[Tuple[bool, InvalidConfig], Tuple[bool, None]]:
    fields = {"PERFORMER": bool, "MODEL_DIR": str, "DEFAULT_MODEL_NAME": str, "VERSION_OVERRIDE": bool,
              "VERSION": str, "HOST": str, "PORT": int, "FILTERED_WORDS": list, "TF_CPP_MIN_LOG_LEVEL": str,
              "UVICORN_WORKERS": str, "MESSAGE_TIMEOUT": int, "CACHE_REQUEST_MAX": int, "MAX_CACHE_STORE": int,
              "LOGGING_LEVEL": str}

    for field, data_type in fields.items():
        if field not in config.keys():
            return False, InvalidConfig(f"{field} is missing from config")
        if not isinstance(config[field], data_type):
            return False, InvalidConfig(f"{field} is not of type {data_type}")
    return True, None
