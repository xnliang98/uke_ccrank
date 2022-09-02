# Unsupervised Keyphrase Extraction
This is code for EMNLP 2021 paper: [Unsupervised Keyphrase Extraction by Jointly Modeling Local and Global Context](https://aclanthology.org/2021.emnlp-main.14/).

*Update 2022-09-02:* We release the data preprocess script and chinese keyphrase extraction code in https://github.com/xnliang98/CKE-ZH.

## requirements
- transformers==3.0.2
- nltk
- pytorch
- tqdm

We employ StanfordCoreNLP Tools to preprocess the data.

## Runing
Step 1: obtain embeddings of candidate phrases and the whole document.
```shell
python src/get_embedding.py --file_path [data_path] --file_name [file_name] --model_name [pretrained model name/path]
```

Step 2: extract keyphrases
```shell
python src/ranker.py [data_path] [model_name]
```

## Comments & TODO
The middle layer representation of BERT model may get better performance.
