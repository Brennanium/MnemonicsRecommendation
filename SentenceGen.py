import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from happytransformer import HappyWordPrediction
import random
from datetime import datetime

happy_wp = HappyWordPrediction()
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
punctuation = ['.', '!', '?']

def gen_sentence(input_mnemonics):

    sentences = []

    # Generates the end of the sentence for each input mnemonic
    print('Encoding.')
    inputs = []
    for i in input_mnemonics:
        inputs.append(tokenizer.encode(i, return_tensors='pt')[0])

    # Input mnemonics get encoded into different token lengths, and the model,
    # needs equal length inputs for all inputs in a batch so we seperate them out
    # into batches of equal token length to speed up the generation
    print('Generating end of sentences.')
    token_lenghts = {len(encoding) for encoding in inputs}
    for lenght in token_lenghts:
        curr_list = [encoding for encoding in inputs if len(encoding) == lenght]
        outputs = model.generate(torch.stack(curr_list), max_length=15, do_sample=True)
        for tensor in outputs:
            sentences.append(tokenizer.decode(tensor, skip_special_tokens=True))

    # Generates the start of each sentence before the mnemonic
    print('Generating beginning of sentences.')
    for i in range(len(sentences)):
        print('Sentence:', i, 'done.')

        # Cuts down the output from the model to just one sentence
        last_idx = len(sentences[i])
        for p in punctuation:
            try:
                last_idx = min(last_idx, sentences[i].index(p))
            except:
                pass
        sentences[i] = sentences[i][:last_idx+1]

        last_token = ''
        for _ in range(random.randrange(0,8)):
            temp = '[MASK] ' + sentences[i]
            result = happy_wp.predict_mask(temp, top_k=5)
            new_token = result[random.randrange(0, 5)].token
            while new_token == '•' or last_token == new_token:
                new_token = result[random.randrange(0, 5)].token
                last_token = new_token
            sentences[i] = new_token + ' ' + sentences[i]

    return sentences
