import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


class TextDataProcessor(object):
    raw_data = None
    train_data_class = []
    train_data_set = []
    classify_data_class = []
    classify_data_set = []
    train_data_feature_matrix = None
    classify_data_feature_matrix = None

    def __init__(self, file_path):
        self.read_raw_data_from_csv(file_path)

    def read_raw_data_from_csv(self, file_path, sep='\t', header=None):
        self.raw_data = pd.read_csv(file_path, sep=sep, header=header)

    def split_dataset(self, train_data_ratio):
        for index in range(len(self.raw_data[1])):
            random_ball = random.random()
            if random_ball < train_data_ratio:
                self.train_data_class.append(self.raw_data[0][index])
                self.train_data_set.append(self.raw_data[1][index])
            else:
                self.classify_data_class.append(self.raw_data[0][index])
                self.classify_data_set.append(self.raw_data[1][index])

    def feature_extract_with_tfid(self):
        vectorizer = TfidfVectorizer(stop_words='english')
        self.train_data_feature_matrix = vectorizer.fit_transform(self.train_data_set)
        self.classify_data_feature_matrix = vectorizer.fit(self.train_data_set).transform(self.classify_data_set)

    def feature_extract_with_count(self):
        vectorizer = CountVectorizer(stop_words='english')
        self.train_data_feature_matrix = vectorizer.fit_transform(self.train_data_set)
        self.classify_data_feature_matrix = vectorizer.fit(self.train_data_set).transform(self.classify_data_set)
