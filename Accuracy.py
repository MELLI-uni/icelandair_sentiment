import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

def accuracy(actual_file, test_file):
    """
    accuracy function displays the confusion matrix and calculates precision, recall, f1, f1 micro-average, f1 macro-average scores, and prints it in a tabular format

    : param actual_file: xlsx file with the truth data
    : param test_file: xlsx file with the predicted data

    : return: N/A
    """
    # Open the two files and convert Sentiment column to list
    df_truth = pd.read_excel(actual_file)
    senti_truth = df_truth['Sentiment'].tolist()
    labels = np.unique(senti_truth)

    df_predict = pd.read_excel(test_file)
    senti_predict = df_predict['Sentiment'].tolist()

    # Generate confusion matrix
    cf_matrix = confusion_matrix(senti_truth, senti_predict, labels=labels)
    df_matrix = pd.DataFrame(cf_matrix, index=labels, columns=labels)

    ax = sns.heatmap(df_matrix, annot=True, cmap='Blues', linecolor='white', cbar='True', xticklabels='auto', yticklabels='auto')
    ax.set(title = "Confusion Matrix",
            xlabel = "Predicted Sentiments",
            ylabel = "Actual Sentiments")
    sns.set(font_scale=0.7)

    plt.show()

    # Calculate precision, recall, accuracy
    precision = precision_score(senti_truth, senti_predict, labels = labels, average = None)
    recall = recall_score(senti_truth, senti_predict, labels = labels, average = None)
    #accuracy = accuracy_score(senti_truth, senti_predict, labels = labels)

    # Calculate f1 score
    f1_gen = f1_score(senti_truth, senti_predict, labels = labels, average = None)
    # Micro average f1 -> calculates positive and negative values globally
    f1_micro = f1_score(senti_truth, senti_predict, labels = labels, average='micro')
    # Macro average f1 -> takes the average of each class's F1 score
    f1_macro = f1_score(senti_truth, senti_predict, labels = labels, average='macro')

    # Compile scores in panda dataframe
    score_compile = np.array([precision, recall, f1_gen])
    f1_average = np.array([f1_micro, f1_macro])
    df_score = pd.DataFrame(data=score_compile, index=['Precision', 'Recall', 'F1'], columns=labels)
    df_average = pd.DataFrame(data=f1_average, index=['F1 Microaverage', 'F1 Macroaverage'], columns=['Scores'])

    # Print dataframe in tabular format
    print(tabulate(df_score, headers='keys', tablefmt='pretty'))
    print(tabulate(df_average, headers='keys', tablefmt='pretty'))
