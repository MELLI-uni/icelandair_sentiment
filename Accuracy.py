# TASK7: Create an accuracy file that returns F1, precision, recall value in a table format when inputted two files (testing and actual)

def accuracy(actual_file, test_file):
    # Precision Score = TP / (FP + TP)
    # Recall Score = TP / (FN + TP)
    # Accuracy Score = (TP + TN) / (TP + FN + TN + FP)
    # F1 Score = 2* Precision Score * Recall Score / (Precision Score + Recall Score)
    print("accuracy test")