from metrics_calculator import MetricsCalculator

class Evaluator:
    def __init__(self):
        self.methods = ["HOG", "R-CNN", "YOLO"]
        self.filters = ["Custom Filter", "AU-GAN", "SID"]
    
    def analyze_metrics(self):
        metrics_table = [["Method/Filter", "Accuracy", "Precision", "Recall", "F1 Score", "IOU", "FPS"]]
        
        for method in self.methods:
            for filter in self.filters:
                true_positives, false_positives, false_negatives, true_negatives = self.calculate_detection_results(method, filter)
                processing_time_per_image = self.calculate_processing_time(method, filter)
                
                metrics_calculator = MetricsCalculator(method)
                accuracy, precision, recall, f1_score, iou, fps = metrics_calculator.calculate_metrics(true_positives, false_positives, false_negatives, true_negatives, processing_time_per_image)
                
                metrics_table.append([f"{method} ({filter})", accuracy, precision, recall, f1_score, iou, fps])
        
        self.print_metrics_table(metrics_table)
    
    def calculate_detection_results(self, method, filter):
        true_positives = 5
        false_positives = 10
        false_negatives = 32
        true_negatives = 4
        return true_positives, false_positives, false_negatives, true_negatives
    
    def calculate_processing_time(self, method, filter):
        processing_time_per_image = 0.02
        return processing_time_per_image
    
    def print_metrics_table(self, metrics_table):
        for row in metrics_table:
            print("|".join(map(str, row)))


def test():
  analyzer = Evaluator()
  analyzer.analyze_metrics()

test()