# encoding: utf-8
import os
import re
import string
import json
import pickle

import nltk
import torch
import numpy as np

from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel

from nltk.corpus import stopwords
stopword_dict = set(stopwords.words('english'))

def print_run_time(func):
    import time
    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)
        print("Current function [%s] run time is %.8f (s)" % (func.__name__, time.time() - local_time))
        return res
    return wrapper

@print_run_time
def get_filepath_list(path):
    g = os.walk(path)
    file_path_list = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            file_path_list.append(os.path.join(path, file_name))
    return file_path_list

@print_run_time
def read_json_by_line(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines  = f.readlines()
    return [json.loads(line.strip()) for line in lines]

def encode_sentence(tokenizer, model, tokens):
    is_split = []
    input_tokens = ['[CLS]']
    for token in tokens:
        tmp = tokenizer.tokenize(token)
        
        if len(input_tokens) + len(tmp) >= 511:
            break
        else:
            input_tokens.extend(tmp)
            is_split.append(len(tmp))
    input_tokens += ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    
    input_ids = torch.LongTensor([input_ids])
    o1, o2, o3 = model(input_ids, output_hidden_states=True)
    bertcls  = o2.squeeze().detach().numpy()
    o1 = o1.squeeze().detach().numpy()
    cls_token = o1[0]
    
    tokens_emb = []
    i = 1
    for j in is_split:
        if j == 1:
            tokens_emb.append(o1[i])
            i += 1
        else:
            tokens_emb.append(sum(o1[i:i+j]) / j)
            # tokens_emb.append(np.max(np.array(o1[i: i+j]), axis=0))
            i += j
        # if i >= len(is_split):
        #     break
    assert len(tokens_emb) == len(is_split)
    return tokens_emb, bertcls, cls_token

def flat_list(l):
    return [x for ll in l for x in ll]

def encode_sentences(file_path, file_name, tokenizer, model, model_name):
    json_list = read_json_by_line(os.path.join(file_path, file_name))

    document_embeddings = []
    for document in tqdm(json_list, desc="Encoding sentences"):
        document_id = document['document_id']

        tokens = flat_list(document['tokens'])
        tokens_emb, bertcls, cls_token = encode_sentence(tokenizer, model, tokens)
        # sentence_embeddings = []
        # tokens_embeddings = []
        # sentence_embeddings = cls_token
        # sentence_embeddings = bertcls
        # for tokens in document['tokens']:
        #     tokens_emb, bertcls, cls_token = encode_sentence(tokenizer, model, tokens)
        #     tokens_embeddings.append(tokens_emb)
        #     # sentence_embeddings.append(bertcls)
        #     sentence_embeddings.append(cls_token)
        document_embeddings.append({
            'document_id': document_id,
            'doc_cls': cls_token,
            'doc_bertcls': bertcls,
            "tokens": tokens_emb
        })

    
    
    with open(os.path.join(file_path, f"test.doclevel.embeddings.{model_name}.pkl"), 'wb') as f:
        pickle.dump(document_embeddings, f)
    return document_embeddings

def extract_candidates(tokens_tagged, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """
    
    GRAMMAR1 = """  NP:
            {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

    GRAMMAR2 = """  NP:
            {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

    GRAMMAR3 = """  NP:
            {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""
    
    np_parser = nltk.RegexpParser(GRAMMAR1)  # Noun phrase parser
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            keyphrase_candidate.append((np, start_end))

        else:
            count += 1

    return keyphrase_candidate

def get_cadidate_embeddings(json_list, document_embeddings):
    document_feats = []
    for document, document_emb in tqdm(zip(json_list, document_embeddings), total=len(json_list)):
        assert document['document_id'] == document_emb['document_id']
        # candidate_phrases = []
        # candidate_phrases_embeddings = []
        sentence = flat_list(document['tokens'])
        sentence_pos = flat_list(document['tokens_pos'])
        sentence_emb = document_emb['tokens']
        
        tokens_tagged = list(zip(sentence, sentence_pos))
        for i, token in enumerate(sentence):
            if token.lower() in stopword_dict:
                tokens_tagged[i] = (token, "IN")
        candidate_phrase = extract_candidates(tokens_tagged)
           
        tmp_embeddings = []
        tmp_candidate_phrase = []
        
        for tmp, (i, j) in candidate_phrase:
            if j<=i:
                continue
            if j >= len(sentence_emb):
                break
            # tmp_embeddings.append(sum(sentence_emb[i:j]) / (j-i))
            tmp_embeddings.append(np.max(np.array(sentence_emb[i:j]), axis=0))
            tmp_candidate_phrase.append(tmp)
        candidate_phrases_embeddings = tmp_embeddings
        candidate_phrases = tmp_candidate_phrase

        document_feats.append({
            'document_id': document['document_id'],
            'tokens': document['tokens'],
            'candidate_phrases': candidate_phrases,
            'candidate_phrases_embeddings': candidate_phrases_embeddings,
            # 'sentence_embeddings': document_emb['doc_bertcls'],
            'sentence_embeddings': document_emb['doc_cls'],
            'keyphrases': document['keyphrases']
        })
    return document_feats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data/DUC2001",
        help="data dir")
    parser.add_argument("--file_name", type=str, default="test.json",
        help="data name with json format")
    parser.add_argument("--model_name", type=str, default="pretrained_models/bert-base-uncased")
    # parser.add_argument("--model_type", type=str, default='bert')

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name)

    
    document_embeddings = encode_sentences(args.file_path, args.file_name, tokenizer, model, args.model_name)
    json_list = read_json_by_line(os.path.join(args.file_path, args.file_name))

    with open(os.path.join(args.file_path, f"test.doclevel.embeddings.{args.model_name}.pkl"), 'rb') as f:
        document_embeddings = pickle.load(f)
    
    document_feats = get_cadidate_embeddings(json_list, document_embeddings)
    with open(os.path.join(args.file_path, f"test.doclevel.feats.{args.model_name}.max.pkl"), 'wb') as f:
        pickle.dump(document_feats, f)


