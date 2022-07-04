from fastapi import FastAPI
from src.utils import *
import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification
from pyserini.search import pysearch

app = FastAPI()

def predict(model, q_text, cands, max_seq_len):
    # Convert list to numpy array
    cands_id = np.array(cands)
    # Empty list for the probability scores of relevancy
    scores = []
    # For each answer in the candidates
    for docid in cands:
        # Map the docid to text
        ans_text = docid_to_text[docid]
        # Create inputs for the model
        encoded_seq = tokenizer.encode_plus(q_text, ans_text,
                                            max_length=max_seq_len,
                                            pad_to_max_length=True,
                                            return_token_type_ids=True,
                                            return_attention_mask = True)

        # Numericalized, padded, clipped seq with special tokens
        input_ids = torch.tensor([encoded_seq['input_ids']]).to(device)
        # Specify question seq and answer seq
        token_type_ids = torch.tensor([encoded_seq['token_type_ids']]).to(device)
        # Sepecify which position is part of the seq which is padded
        att_mask = torch.tensor([encoded_seq['attention_mask']]).to(device)
        # Don't calculate gradients
        with torch.no_grad():
            # Forward pass, calculate logit predictions for each QA pair
            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=att_mask)
        # Get the predictions
        logits = outputs[0]
        # Apply activation function
        pred = softmax(logits, dim=1)
        # Move logits and labels to CPU
        pred = pred.detach().cpu().numpy()
        # Append relevant scores to list (where label = 1)
        scores.append(pred[:,1][0])
        # Get the indices of the sorted similarity scores
        sorted_index = np.argsort(scores)[::-1]
        # Get the list of docid from the sorted indices
        ranked_ans = list(cands_id[sorted_index])

    return ranked_ans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

config = {'bert_model_name': 'bert-qa',
          'max_seq_len': 512,
          'batch_size': 16,
          'learning_rate': 3e-6,
          'weight_decay': 0.01,
          'n_epochs': 2,
          'num_warmup_steps': 10000}
# Dictionary mapping docid and qid to raw text
docid_to_text = load_pickle('data/id_to_text/docid_to_text.pickle')
qid_to_text = load_pickle('data/id_to_text/qid_to_text.pickle')

# Load the BERT tokenizer.
print('\nLoading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Download pre-trained BERT model
# The model was converted from TensorFlow to PyTorch format
get_model(config['bert_model_name'])

if config['bert_model_name'] == 'bert-base-uncased':
    model_path = config['bert_model_name']
else:
    model_path = "model/" + config['bert_model_name']

# Load BertForSequenceClassification - pretrained BERT model
# with a single linear classification layer on top
model = BertForSequenceClassification.from_pretrained(model_path, cache_dir=None, num_labels=2)

model.to(device)

model_name = 'finbert-qa'

checkpoint = get_trained_model(model_name)

trained_model_path = "model/trained/" + model_name + "/" + checkpoint

# Load model

model.load_state_dict(torch.load(trained_model_path,map_location=torch.device('cpu')), strict=False)
k = 10
# Lucene index
FIQA_INDEX = "retriever/lucene-index-fiqa"

# Retriever using Pyserini
searcher = pysearch.SimpleSearcher(FIQA_INDEX)
print('Ready...')

@app.get("/search/{userquery}")
async def read_query(userquery: str):
    query = userquery
    print(query)
    # Retrieve top-50 answer candidates
    hits = searcher.search(query, k=50)
    cands = []

    for i in range(0, len(hits)):
        cands.append(int(hits[i].docid))


    import time
    start_time = time.time()
    # Re-rank candidates
    rank = predict(model, query, cands, config['max_seq_len'])
    print("--- %s seconds ---" % (time.time() - start_time))

    # Print the Top-k answers
    k = 1

    print("Query:\n\t{}\n".format(query))
    print("Top-{} Answers: \n".format(k))
    for i in range(0, k):
        print("{}.\t{}\n".format(i + 1, docid_to_text[rank[i]]))

    return {"Answer": "{}".format(docid_to_text[rank[i]])}