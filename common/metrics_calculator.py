class MetricsCalculator:
    def __init__(self, method):
        self.method = method
    
    def calculate_metrics(self, true_positives, false_positives, false_negatives, true_negatives, processing_time_per_image):
        accuracy = self.calculate_accuracy(true_positives, true_negatives, false_positives, false_negatives)
        precision = self.calculate_precision(true_positives, false_positives)
        recall = self.calculate_recall(true_positives, false_negatives)
        f1_score = self.calculate_f1_score(precision, recall)
        iou = self.calculate_iou(true_positives, false_positives, false_negatives)
        fps = self.calculate_fps(processing_time_per_image)
        return accuracy, precision, recall, f1_score, iou, fps
    
    def calculate_accuracy(self, true_positives, true_negatives, false_positives, false_negatives):
        total = true_positives + true_negatives + false_positives + false_negatives
        accuracy = (true_positives + true_negatives) / total
        return accuracy
    
    def calculate_precision(self, true_positives, false_positives):
        if true_positives + false_positives == 0:
            return 0
        precision = true_positives / (true_positives + false_positives)
        return precision
    
    def calculate_recall(self, true_positives, false_negatives):
        if true_positives + false_negatives == 0:
            return 0
        recall = true_positives / (true_positives + false_negatives)
        return recall
    
    def calculate_f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
    def calculate_iou(self, true_positives, false_positives, false_negatives):
        if true_positives + false_positives + false_negatives == 0:
            return 0
        iou = true_positives / (true_positives + false_positives + false_negatives)
        return iou
    
    def calculate_fps(self, processing_time_per_image):
        if processing_time_per_image == 0:
            return 0
        fps = 1 / processing_time_per_image
        return fps