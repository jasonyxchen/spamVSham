from __future__ import division


class Statistics(object):
    @classmethod
    def statistics_classifier_ratio(cls, target_classes, classify_classes):
        successful_count = 0
        if len(target_classes) != len(classify_classes):
            raise AssertionError('Classes count is not matched')
        for index in range(len(target_classes)):
            if target_classes[index] == classify_classes[index]:
                successful_count += 1
        all_count = len(target_classes)
        successful_ratio = successful_count / all_count
        return successful_ratio
