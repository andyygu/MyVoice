import pandas as pd 
import csv
import nltk
from nltk import FreqDist
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import random
from nltk import classify
from nltk import NaiveBayesClassifier

not_worried_data = pd.read_csv('not_worried.csv')
worried_data = pd.read_csv('worried.csv')

not_worried_tokens = []
worried_tokens = []

prompt = "Are you worried about coronavirus? Why or why not?"

#lemmatize function
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

#tokenize and lemmatize function
def tokenize_and_lemmatize(response):
    words = response.split()
    response_tokens = []
    for word in words:
        if word not in response_tokens: 
            response_tokens.append(word)
    return lemmatize_sentence(response_tokens)
        

#tokenize call - tokenizing the data
for index, row in not_worried_data.iterrows():
    not_worried_tokens.append(tokenize_and_lemmatize(row[prompt]))

for index, row in worried_data.iterrows():
    worried_tokens.append(tokenize_and_lemmatize(row[prompt]))

#print(not_worried_tokens)

# determine word density 
def get_all_words(all_tokens):
    for tokens in all_tokens:
        for token in tokens:
            yield token

all_not_worried_words = get_all_words(not_worried_tokens)
all_worried_words = get_all_words(worried_tokens)

freq_dist_pos = FreqDist(all_not_worried_words)
#print(freq_dist_pos.most_common(10))



#not_worried_tokens_lemmatized = []
#worried_tokens_lemmatized = []

#lemmatize function call
# for index, row in not_worried_data.iterrows()
#     not_worried_tokens_lemmatized.append(lemmatize_se


#print(lemmatize_sentence(not_worried_tokens[1]))

########################## PREPARING MODEL CODE ###################################################

def get_responses_for_model(tokens_list):
    for tokens in tokens_list:
        yield dict([token, True] for token in tokens)

not_worried_tokens_for_model = get_responses_for_model(not_worried_tokens)
worried_tokens_for_model = get_responses_for_model(worried_tokens)


# Splitting the data into train and test

not_worried_dataset = [(response_dict, "Not_worried")
                        for response_dict in not_worried_tokens_for_model]

worried_dataset = [(response_dict, "Worried")
                    for response_dict in worried_tokens_for_model]

dataset = not_worried_dataset + worried_dataset
random.shuffle(dataset)

train_data = dataset[:200]
test_data = dataset[200:]

#print(train_data)

########################## BUILDING/TESTING MODEL CODE ###################################################

classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))



