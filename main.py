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
import uvicorn
import contextlib
import threading
import time
import json
import os
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


# From: https://github.com/encode/uvicorn/issues/742#issuecomment-674411676
class Server(uvicorn.Server):
    def __init__(self, config):
        super().__init__(config)
        self.thread = threading.Thread(target=self.run)

    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        self.thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            self.thread.join()

    def start(self):
        self.thread.start()

    def stop(self):
        self.should_exit = True
        self.thread.join()


def on_modified(_):
    global server, Config, flag
    if not flag:
        flag = True
        server.stop()
        server = Server(config=Config)
        server.start()
    else:
        flag = False


if __name__ == "__main__":
    api_config = json.load(fp=open("api_config.json", "rb"))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = api_config['TF_CPP_MIN_LOG_LEVEL']

    flag = False
    Config = uvicorn.Config("gavin:api", host=api_config['HOST'], port=api_config['PORT'], log_level=api_config['LOG_LEVEL'], reload=True, workers=api_config["UVICORN_WORKERS"])
    server = Server(config=Config)
    server.start()

    path = '.'
    event_handler = PatternMatchingEventHandler(patterns=['gavin.py', 'utils.py', "api_config.json", "chat_bot.py"])
    event_handler.on_modified = on_modified
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while observer.is_alive():
            observer.join(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
