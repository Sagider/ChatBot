# This project is from the Tech With Tim YouTube channel

import json
import pickle
import random

import nltk
import numpy
import tensorflow
import tflearn
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []  # The actual words will be stored here
    labels = []  # The unique tag names will bes stored here
    docs_x = []  # The list of words/strings for each pattern will be stored here
    docs_y = []  # The tag names will be stored here but used for training purpose, the names needn't be unique

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)  # This basically makes a list of each word in the pattern
            words.extend(wrds)  # We add all the words from all patterns in the entire intents file to words
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])  # This ensures uniqueness of the elements

    words = [stemmer.stem(w.lower()) for w in words if
             w != '?']  # This removes the morphological affixes leaving the stem
    words = sorted(list(set(words)))  # This step deletes duplicate words in words and sorts them in alphabetical order

    labels = sorted(labels)  # Sorts labels in alphabetical order

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    '''The enumerate () method adds a counter to an iterable and returns it in the form of an enumerating object. 
    This enumerated object can then be used directly for loops or converted into a list of tuples using the list() 
    function. (geeksforgeeks)'''

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")

except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
'''
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")'''


def bag_of_words(s, words):
    pbag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                pbag[i] = 1
    return numpy.array(pbag)

def chat():
    print("talk (type quit to stop)")
    while True:
        inp = input("you: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]

        print(random.choice(responses))

chat()

