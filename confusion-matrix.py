import csv
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced


def unite_arrays(directory):
    print("Подготовка массива из файлов в папке " + directory + " ...")
    files = os.listdir(directory)
    files_names = list(filter(lambda x: x.endswith('.csv'), files))
    files_names = sorted(files_names)
    total = len(files_names)
    print("В папке обнаружено " + str(total) + " файлов!")

    array = []
    for file in os.listdir(directory):
        print("Чтение файла " + file + "...")
        with open(directory + file) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                array += row
    return array


print("Построение списка верных ответов...")
y_true = unite_arrays('./test_output/')
print(y_true)
print("Построение списка предсказанных ответов...")
y_pred = unite_arrays('./predict_output/')
print(y_pred)

print("Построение матрицы спутанности...")
confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
print(confusion_matrix)
print(classification_report_imbalanced(y_true, y_pred))

print("Запись матрицы спутанности в файл...")
plt.figure(figsize=(100, 100))
sns.heatmap(confusion_matrix, linewidths=0.1, robust=True, annot=True, fmt=".1f")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')
plt.close()

print("Вывод матрицы спутанности на экран...")
plt.figure(figsize=(100, 100))
sns.heatmap(confusion_matrix, linewidths=0.1, robust=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
