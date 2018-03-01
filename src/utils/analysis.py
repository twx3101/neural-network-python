def confusionMatrix(T, x2, binary_targets, no_of_classes):
    """Generates and outputs a confusion matrix for the lists of trees T, data x2, correct classifications, binary_targets and the number of classes no_of_classes."""

    confusion_matrix = np.zeros((no_of_classes,no_of_classes))

    prediction_array = testTrees(T,x2) #change this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    for i in range(no_of_classes):
        for j in range(no_of_classes):
            for k in range(len(binary_targets)):
                if binary_targets[k] == j+1 and prediction_array[k] == i+1:
                    confusion_matrix[i][j] += 1

    return confusion_matrix


def averageRecall(confusion_matrix, class_number):
    """returns average recall for the class"""

    total_actual = 0
    for row in confusion_matrix:
        total_actual += row[class_number - 1]

    true_positives = confusion_matrix[class_number - 1][class_number - 1]

    return float(true_positives)/total_actual


def precisionRate(confusion_matrix, class_number):
    """returns precision rate for the class"""
    # precision = True Positive/ (True Positive + False Positive)
    # '' in-row sum '''
    total_predicted = 0
    for i in confusion_matrix[class_number - 1]:
        total_predicted += i

    true_positives = confusion_matrix[class_number - 1][class_number - 1]

    return float(true_positives)/total_predicted

def f1(precision, recall):
    """calculates and returns the f1 measure using the precision and recall"""
    if precision == 0 and recall == 0:
        return 0
    return (2 * float((precision * recall))/(precision + recall))


def classificationRate(confusion_matrix, no_of_classes):
    """calculates and return the classification rate for one class."""
    total = 0
    for row in confusion_matrix:
        for cell in row:
            total += cell
    total_true = 0
    i = 0
    while i < no_of_classes:
        total_true += confusion_matrix[i][i]
        i += 1
    return float(total_true) / total
