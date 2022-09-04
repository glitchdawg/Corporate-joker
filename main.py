# Preliminaries
import numpy as np
import pandas as pd
import streamlit as st
import random
import time

# transformers
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

# Pytorch
import torch
import torch.nn as nn

# warnings
import warnings

warnings.filterwarnings('ignore')

Tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')


# Helper Function
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
device = torch.device('cpu')


#@st.experimental_singleton(func=None, show_spinner=True, suppress_st_warning=True)
def model1_load():
    # Model Loading
    # model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<soq>', 'sep_token': '<eoq>'}
    num_added_toks = Tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens')
    model.resize_token_embeddings(len(Tokenizer))

    # loading Model state
    models_path = "trained_models/Sentence Completion.pt"  # ADD PATH TO YOUR SAVED MODEL HERE
    model.load_state_dict(torch.load(models_path, map_location='cpu'))

    # device = torch.device('cpu')  # Selecting Device
    model.to(device)


def predict1(start_of_joke, length_of_joke=96, number_of_jokes=2):
    joke_num = 0
    model.eval()
    with torch.no_grad():
        for joke_idx in range(number_of_jokes):

            joke_finished = False

            cur_ids = torch.tensor(Tokenizer.encode(start_of_joke)).unsqueeze(0).to(device)

            for i in range(length_of_joke):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0, -1],
                                               dim=0)  # Take the first(from only one in this case) batch and the last predicted embedding
                if i < 3:
                    n = 20
                else:
                    n = 3
                next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(),
                                                n=n)  # Randomly(from the topN probability distribution) select the next word
                cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id],
                                    dim=1)  # Add the last word to the running sequence

                if next_token_id in Tokenizer.encode('<|endoftext|>'):
                    joke_finished = True
                    break

            if joke_finished:
                joke_num = joke_num + 1

                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = Tokenizer.decode(output_list)
                output_text=output_text.replace('<|endoftext|>', '')
                output_text=output_text.replace('<PAD>', '')
                output_text=output_text.replace('<soq>', '')
                output_text=output_text.replace('<eoq>', '')
                st.write(output_text + '\n')
def model2_load():
    special_tokens_dict = {'pad_token': '<PAD>'}
    num_added_toks = config.Tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens')
    model.resize_token_embeddings(len(config.Tokenizer))

    #loading Model State
    models_path = "trained_models/Joke Generator.pt" # ADD PATH TO YOUR SAVED MODEL HERE
    model.load_state_dict(torch.load(models_path, map_location='cpu'))


    model.to(device)

def predict2(length_of_joke,number_of_jokes):
    joke_num = 0
    model.eval()
    with torch.no_grad():
        for joke_idx in range(number_of_jokes):
        
            joke_finished = False

            cur_ids = torch.tensor(config.Tokenizer.encode('JOKE')).unsqueeze(0).to(device)

            for i in range(length_of_joke):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
                if i < 3:
                    n = 20
                else:
                    n = 3
                next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

                if next_token_id in config.Tokenizer.encode('<|endoftext|>'):
                    joke_finished = True
                    break

            
            if joke_finished:
                
                joke_num = joke_num + 1
                
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = config.Tokenizer.decode(output_list)

                print(output_text+'\n')




bitlist = []
for x in range(128, 200):
    if x % 2 == 0:
        bitlist.append(x)
bits = random.choice(bitlist)

st.title("Corporate Joker")
e = st.text_input("Enter a Joke", "Some random joke")
button = st.button("Generate")
if e:
    if button:
        model1_load()
        predict1(e, bits, 1)

else: 
    if button:
        model2_load()
        predict2(bits, 1)
