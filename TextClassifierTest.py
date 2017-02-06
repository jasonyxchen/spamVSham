import sys
from TextDataProcessor import TextDataProcessor
from TextDataClassifier import TextDataClassifier
from Statistics import Statistics


class TextClassifierTest(object):
    file_path = None
    train_data_ratio = 0

    def __init__(self, file_path, train_data_ratio):
        self.file_path = file_path
        self.train_data_ratio = train_data_ratio

    def process(self):
        data_processor = TextDataProcessor(self.file_path)
        data_processor.split_dataset(float(self.train_data_ratio))
        data_processor.feature_extract_with_tfid()
        svm_data_classifier = TextDataClassifier('liner_svm')
        nb_data_classifier = TextDataClassifier('mn_bayes')
        rbf_svm_data_classifier = TextDataClassifier('rbf_svm')
        svm_data_classifier.train(data_processor.train_data_class, data_processor.train_data_feature_matrix)
        nb_data_classifier.train(data_processor.train_data_class, data_processor.train_data_feature_matrix)
        rbf_svm_data_classifier.train(data_processor.train_data_class, data_processor.train_data_feature_matrix)
        svm_classified_vector = svm_data_classifier.classify(data_processor.classify_data_feature_matrix)
        nb_classified_vector = nb_data_classifier.classify(data_processor.classify_data_feature_matrix)
        rbf_svm_classified_vector = rbf_svm_data_classifier.classify(data_processor.classify_data_feature_matrix)
        svm_ratio = Statistics.statistics_classifier_ratio(data_processor.classify_data_class, svm_classified_vector)
        nb_ratio = Statistics.statistics_classifier_ratio(data_processor.classify_data_class, nb_classified_vector)
        rbf_ratio = Statistics.statistics_classifier_ratio(data_processor.classify_data_class, rbf_svm_classified_vector)
        print('SVM classifier with liner kernel function successful ratio: %f\n' % svm_ratio)
        print('SVM classifier with RBF kernel function successful ratio: %f\n' % rbf_ratio)
        print('Multinomial Naive Bayes classifier successful ratio: %f\n' % nb_ratio)

if __name__ == "__main__":
    tester = TextClassifierTest(sys.argv[1], sys.argv[2])
    tester.process()

