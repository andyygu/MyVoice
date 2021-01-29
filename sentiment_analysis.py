import pandas as pd 
import csv
import nltk
from nltk import FreqDist
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import random
from nltk import classify
from nltk import NaiveBayesClassifier

class SAClassifier:
    def __init__(self, not_worried_file, worried_file):
        self.not_worried_file = not_worried_file
        self.worried_file = worried_file

    def get_responses_for_model(self, tokens_list):
        for tokens in tokens_list:
            yield dict([token, True] for token in tokens)

    #lemmatize function
    def lemmatize_sentence(self, tokens):
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
    def tokenize_and_lemmatize(self, response):
        words = response.split()
        response_tokens = []
        for word in words:
            if word not in response_tokens: 
                response_tokens.append(word)
        return self.lemmatize_sentence(response_tokens)

    # determine word density 
    def get_all_words(self, all_tokens):
        for tokens in all_tokens:
            for token in tokens:
                yield token
    
    def is_worried(self, response):
        response = self.tokenize_and_lemmatize(response)
        # print("ASD;LKJF;ALSKDFJ;A LDKJF;LASKDJF;LASDKJF ;ASKLDJ")
        # print(self.classifier.classify(dict([token, True] for token in response)))
        # print("asdf;laksdjf;alkjdf;adlskjf;lsadkjf")
        return self.classifier.classify(dict([token, True] for token in response))

    def make_model(self):
        not_worried_data = pd.read_csv(self.not_worried_file)
        worried_data = pd.read_csv(self.worried_file)

        not_worried_tokens = []
        worried_tokens = []

        prompt = "Are you worried about coronavirus? Why or why not?"
        
        #tokenize call - tokenizing the data
        for index, row in not_worried_data.iterrows():
            not_worried_tokens.append(self.tokenize_and_lemmatize(row[prompt]))

        for index, row in worried_data.iterrows():
            worried_tokens.append(self.tokenize_and_lemmatize(row[prompt]))

        
        all_not_worried_words = self.get_all_words(not_worried_tokens)
        all_worried_words = self.get_all_words(worried_tokens)

        freq_dist_pos = FreqDist(all_not_worried_words)

        not_worried_tokens_for_model = self.get_responses_for_model(not_worried_tokens)
        worried_tokens_for_model = self.get_responses_for_model(worried_tokens)

        # Splitting the data into train and test

        not_worried_dataset = [(response_dict, "Not_worried")
                                for response_dict in not_worried_tokens_for_model]

        worried_dataset = [(response_dict, "Worried")
                            for response_dict in worried_tokens_for_model]

        dataset = not_worried_dataset + worried_dataset
        random.shuffle(dataset)

        train_data = dataset[:200]
        test_data = dataset[200:]

        ########################## BUILDING/TESTING MODEL CODE ###################################################

        self.classifier = NaiveBayesClassifier.train(train_data)
    
        print("Accuracy is:", classify.accuracy(self.classifier, test_data))

        print(self.classifier.show_most_informative_features(10))

    