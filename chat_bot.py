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
import json
import os
import glob
import asyncio
import logging
from packaging import version
from utils import predict, load_model, preprocess_sentence, tf_version


INFO = "INFO:     "
WARN = "WARN:     "
ERROR = "ERROR:     "


class ChatBotTF:
    def __init__(self, start_token, end_token, tokenizer, max_len, model, model_name, hparams, config_file_path, logger):
        self.logger = logger
        self.config = json.load(fp=open(config_file_path, "rb"))
        self.s_token = start_token
        self.e_token = end_token
        self.tokenizer = tokenizer
        self.max_length = max_len
        self.model = model
        self.ModelName = model_name
        fields = ["Samples", "Max Length Of Words", "Batch Size", "Buffer Size", "Layers", "d_model", "Heads", "Units",
                  "Dropout", "Vocab Size"]
        self.hparams_dict = {}
        for i, hparam in enumerate(hparams[:-1]):  # use -1 to get rid of the last value returned by load_model
            self.hparams_dict[fields[i]] = hparam
        self.swear_words = self.config["FILTERED_WORDS"]
        self.INFO = INFO
        self.WARN = WARN
        self.ERROR = ERROR

    @classmethod
    def load(cls, config_file_path="api_config.json"):
        logger = logging.getLogger("ChatBotTF.error")
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
        logger.info(INFO + "Using Tensorflow version: {}")
        model_config = json.load(fp=open(config_file_path, "rb"))
        if not model_config['VERSION_OVERRIDE']:
            model_path = os.path.join(model_config['MODEL_DIR'], model_config['DEFAULT_MODEL_NAME'])
        elif model_config['VERSION_OVERRIDE'] and model_config['VERSION'] == "" or None:
            model_paths = glob.glob(os.path.join(model_config['MODEL_DIR'], "*"))
            versions = {}
            for model_p in model_paths:
                f = open(os.path.join(model_p, "version.txt"), "r")
                versions[version.parse(f.readlines()[0])] = model_p
                f.close()

            model_path = versions[max(versions.keys())]
        elif model_config['VERSION_OVERRIDE'] and model_config['VERSION'] != "" or None:
            model_paths = glob.glob(os.path.join(model_config['MODEL_DIR'], "*"))
            versions = {}
            for model_p in model_paths:
                f = open(os.path.join(model_p, "version.txt"), "r")
                versions[version.parse(f.readlines()[0])] = model_p
                f.close()
            model_path = versions[version.parse(model_config['VERSION'])]

        else:
            message = "Invalid Config. Config: {}".format(model_config)
            logger.error(ERROR + message)
            raise Exception("Invalid Config.")
        message = "Loading Model: {}".format(model_path)
        logger.info(INFO + message)
        start_token, end_token, tokenizer, max_len, model, model_name, hparams = load_model(model_path)
        return_cls = cls(start_token, end_token, tokenizer, max_len, model, model_name, hparams, config_file_path, logger)
        # run a predict once to load extra tf modules
        asyncio.ensure_future(return_cls.predict_msg("test"))
        return return_cls

    async def predict_msg(self, msg):
        return await predict(preprocess_sentence(msg), self.tokenizer, self.swear_words, self.s_token, self.e_token, self.max_length, self.model)
