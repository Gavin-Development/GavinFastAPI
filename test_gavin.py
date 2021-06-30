import json

from fastapi.testclient import TestClient
from gavin import api, ChatBot


client = TestClient(api)


def test_read_hparams():
    response = client.get('/chat_bot/hparams')
    assert response.status_code == 200
    hparams = ChatBot.get_hparams()
    hparams['TOKENIZER'] = f"Tokenizer Object. Vocab_Size: {ChatBot.vocab_size}"
    assert response.json() == hparams


def test_read_model_name():
    response = client.get('/chat_bot/model_name')
    assert response.status_code == 200
    assert response.json() == {'ModelName': ChatBot.name}


def test_read_config():
    response = client.get('/config')
    fp = open('api_config.json', 'rb')
    config = json.load(fp)
    fp.close()
    assert response.status_code == 200
    assert response.json() == config


def test_post_message():
    message = {"data": "Hello."}
    response = client.post('/chat_bot/', json=message)
    assert response.status_code == 200
    assert response.json() == {'message': ChatBot.predict(message["data"])}
