# Llama2-Medical-Chatbot
This version of Medical Chatbot utilizes use of [FAISS](https://github.com/facebookresearch/faiss) library by Meta for vector storage and uses Meta's LLM [LLAMA-2](https://llama.meta.com/llama2/).


The UI is made using [Chainlit](https://github.com/Chainlit/chainlit?tab=readme-ov-file).


## Create a virtual env (ubuntu)
``` bash
    python3 -m venv env
```

## Activate the env
``` bash
    source env/bin/activate
```

## Install dependencies

```bash
    pip install -r requirements.txt
```
## Install the model

Can download the model from [here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)

## Run the app

```bash
    chainlit run app.py -w
```
