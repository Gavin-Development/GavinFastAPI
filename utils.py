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

import re
import tensorflow_datasets as tfds

from GavinBackend.models import Transformer, tf
from GavinBackend.functions import evaluate
from GavinBackend.preprocessing.text import preprocess_sentence

Config = tf.compat.v1.ConfigProto()
Config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=Config)
tf.compat.v1.set_random_seed(1234)

tf_version = tf.__version__


def load_model(base_path):
    save_path = base_path
    checkpoint_path = f"{base_path}/cp.ckpt"
    with open(f"{save_path}/values/hparams.txt", "r", encoding="utf8") as f:
        lines = f.readlines()
        formatted = []
        for line in lines:
            formatted.append(line.replace("\n", ""))
        MAX_SAMPLES = int(formatted[0])
        ModelName = formatted[1]
        MAX_LENGTH = int(formatted[2])
        BATCH_SIZE = int(formatted[3])
        BUFFER_SIZE = int(formatted[4])
        NUM_LAYERS = int(formatted[5])
        D_MODEL = int(formatted[6])
        NUM_HEADS = int(formatted[7])
        UNITS = int(formatted[8])
        DROPOUT = float(formatted[9])
        VOCAB_SIZE = int(formatted[10])
        TARGET_VOCAB_SIZE = int(formatted[11])
        hparams = [MAX_SAMPLES, MAX_LENGTH, BATCH_SIZE, BUFFER_SIZE, NUM_LAYERS, D_MODEL, NUM_HEADS, UNITS, DROPOUT,
                   VOCAB_SIZE, TARGET_VOCAB_SIZE]
        f.close()

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(f"{save_path}/tokenizer/vocabTokenizer")

    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    # Vocabulary size plus start and end token
    # print(f"Vocab size: {VOCAB_SIZE}")

    base = Transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    model = base.return_model()

    model.load_weights(checkpoint_path).expect_partial()
    return START_TOKEN, END_TOKEN, tokenizer, MAX_LENGTH, model, ModelName, hparams


def checkSwear(sentence, swearwords):
    listSentence = sentence.split()
    listSentenceClean = []
    for word in range(len(listSentence)):
        stars = []
        word_regex = re.sub(r"[^a-zA-Z]+", "", listSentence[word].lower())
        if word_regex in swearwords:
            for x in range(5):
                stars.append("#")
            stars = "".join(stars)
            listSentenceClean.append(stars)
        else:
            listSentenceClean.append(listSentence[word])
    listSentenceClean = " ".join(listSentenceClean)
    return listSentenceClean


async def predict(sentence, tokenizer, swear_words, start_token, end_token, max_len, model):
    prediction = evaluate(sentence, model, max_len, start_token, end_token, tokenizer)

    predicated_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
    predicated_sentence = checkSwear(predicated_sentence, swear_words)

    return predicated_sentence
