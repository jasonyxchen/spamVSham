from sklearn.naive_bayes import MultinomialNB
from sklearn import svm


class TextDataClassifier(object):
    clf = None

    def __init__(self, machine_name):
        if machine_name == 'mn_bayes':
            self.clf = MultinomialNB()
        elif machine_name == 'liner_svm':
            self.clf = svm.SVC(kernel='linear')
        elif machine_name == 'rbf_svm':
            self.clf = svm.SVC(kernel='rbf')
        else:
            raise AttributeError('Unsupported machine type')

    def train(self, train_data_class, train_data_matrix):
        self.clf.fit(train_data_matrix, train_data_class)

    def classify(self, classifier_data_matrix):
        class_data = self.clf.predict(classifier_data_matrix)
        return class_data
