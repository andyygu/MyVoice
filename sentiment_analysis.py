import pandas as pd 
import csv
import nltk
from nltk import FreqDist

not_worried_data = pd.read_csv('not_worried.csv')
worried_data = pd.read_csv('worried.csv')

not_worried_tokens = []
worried_tokens = []

prompt = "Are you worried about coronavirus? Why or why not?"

#tokenize function
def tokenize(response):
    words = response.split()
    response_tokens = []
    for word in words:
        if word not in response_tokens: 
            response_tokens.append(word)
    return response_tokens
        

#overall for loop for not worried
for index, row in not_worried_data.iterrows():
    not_worried_tokens.append(tokenize(row[prompt]))

for index, row in worried_data.iterrows():
    worried_tokens.append(tokenize(row[prompt]))

#print(not_worried_tokens)

# generator function
def get_all_words(not_worried_tokens):
    for tokens in not_worried_tokens:
        for token in tokens:
            yield token

all_not_worried_words = get_all_words(not_worried_tokens)

freq_dist_pos = FreqDist(all_not_worried_words)
print(freq_dist_pos.most_common(10))
    

