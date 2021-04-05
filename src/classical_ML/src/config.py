import os
import time
import en_core_web_sm
import csv
import pickle
import string
import json

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from statistics import mean
from sklearn.model_selection import cross_val_score
from scipy import sparse

nlp = en_core_web_sm.load()

ARG_EXTRACTION_ROOT_DIR = os.path.abspath(os.getcwd())
