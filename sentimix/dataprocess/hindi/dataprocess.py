from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging
import re
import nltk
import gensim
import pickle

# import HTMLParser
import itertools
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict

# Read data from files
train = pd.read_csv("./data/get_train.tsv", header=0, delimiter="\t", quoting=3)


# test = pd.read_csv("./data_sentimix/Test_last_trans.tsv", header=0, delimiter="\t", quoting=3)


def review_to_wordlist(review, remove_stopwords=False):
    clean_sentence = []
    review = review.lower()
    review = review.replace("# ", "#")
    review = review.replace("@ ", "@")
    review = review.replace(" _ ", "_")
    review = review.replace(" __ ", "")
    review = review.replace("__ ", "__")
    review = review.replace("_ ", "_")
    review = review.replace(' ’ s ', ' is ')
    review = review.replace(' ’ m ', ' am ')
    review = review.replace(' ’ re ', ' are ')
    review = review.replace("’ ll", 'will')
    review = review.replace("i'm", 'i am')
    review = review.replace("you'll", 'you will')
    review = review.replace("don't", 'do not')
    review = review.replace("can't", "can not")
    review = review.replace("it's", "it is")
    review = review.replace("she's", "she is")
    review = review.replace("let's", "let us")
    review = review.replace("i'll", "i will")
    review = review.replace("haven't", "have not")
    review = review.replace("doesn't", "does not")
    review = review.replace("he's", "he is")
    review = review.replace("doesn ’ t", "does not")
    review = review.replace("didn ’ t", "did not")
    review = review.replace("i ’ ve", "i have")
    review = review.replace("we'll", "we will")
    review = review.replace("i ’ d", "i had")
    review = review.replace("won ’ t", "would not")
    review = review.replace("we ’ ve", "we have")
    review = review.replace("you ’ ve", "you are")
    review = review.replace("ain ’ t", "are not")
    review = review.replace("y ’ all", "you and all")
    review = review.replace("couldn ’ t", "could not")
    review = review.replace("haven ’ t", "have not")
    review = review.replace("aren't", "are not")
    review = review.replace("you ’ d", "you had")
    review = review.replace("that's", "that is")
    review = review.replace("wasn't", "was not")
    review = review.replace("he'll", "he will")
    review = review.replace("ma ’ am", 'madam')
    review = review.replace("ma'am ", "madam")
    review = review.replace("they ’ ve", "they have")
    review = review.replace('don ’ t', 'do not')
    review = review.replace('can ’ t', 'can not')
    review = review.replace('isn ’ t', 'is not')
    review = review.replace("b'day", 'birthday')
    review = review.replace("I've", 'I have')
    review = review.replace("didn't", "did not")
    review = review.replace("u're", "you are")
    review = review.replace("What's", 'what is')
    review = review.replace("you're", 'you are')
    review = review.replace("You're", 'you are')
    review = review.replace("I'm", 'I am')
    review = review.replace("isn't", "is not")
    review = review.replace(" ___", "___ ")
    review = review.replace("won't", 'will not')
    review = review.replace('can ’ t', 'can not')
    review = review.replace('I ’ ll ', 'I will')
    review = review.replace("we ’ ll", 'we will')
    review = review.replace("didn ’ t", 'did not')
    review = review.replace(" u ", ' you ')
    review = review.replace("wasn ’ t", 'was not')
    review = review.replace(' ’ s ', ' is ')
    review = review.replace(' ’ m ', ' am ')
    review = review.replace(' ’ re ', ' are ')
    review = review.replace("’ ll",'will')
    review = review.replace('don ’ t', 'do not')
    review = review.replace('can ’ t', 'can not')
    review = review.replace('isn ’ t', 'is not')
    review = review.replace("I've", 'I have')
    review = review.replace("What's", 'what is')
    review = review.replace("you're", 'you are')

    review = review.replace("You're", 'you are')
    review = review.replace("I'm", 'I am')
    review = review.replace("won't", 'will not')
    review = review.replace('can ’ t', 'can not')
    review = review.replace("we ’ ll", 'we will')
    review = review.replace("didn ’ t", 'did not')
    review = review.replace(" u ", ' you ')
    review = review.replace("wasn ’ t", 'was not')

    # review = review.replace(' ’ re', 'are')
    review = review.replace('+', 'and')
    review_text = BeautifulSoup(review, "lxml").get_text()
    # review_text = _slang_loopup(review_text)
    review_text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(review_text))
    # print(review_text)
    review_text = review_text.split()
    for i in review_text:
        if i.startswith("@"):
            i = 'user'
            i.startswith("")
        if i.startswith("https"):
            continue
        if i.startswith("RT"):
            continue
        else:
            clean_sentence.append(i)
    review_text = ' '.join(str(i) for i in clean_sentence)

    review_text = re.sub("[^a-zA-Zn?!.]", " ", review_text)

    words = review_text.lower().split()
    orig_rev = ' '.join(words).lower()

    return (orig_rev)


if __name__ == '__main__':
    text = ''
    clean_train_reviews = []
    for i, review in enumerate(train["sentence"]):
        # if review == float("Nan"):
        #     print('xxx',review)
        text = review_to_wordlist(review, remove_stopwords=False)
        y = train['label'][i]
        text += '\t' + y
        clean_train_reviews.append(text)
    print(clean_train_reviews)
    with open("train-spanglish.tsv", 'w', encoding='utf8') as file_obj:
        for i in clean_train_reviews:
            file_obj.write(i + '\n')
