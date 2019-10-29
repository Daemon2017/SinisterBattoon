import csv
import os

directory = './train_output/'
files = os.listdir(directory)

files_names = sorted(list(filter(lambda x: x.endswith('.csv'), files)))
total_number_of_files = len(files_names)

total_number_of_values = 0
total_number_of_zero_values = 0
for file in os.listdir(directory):
    print("Чтение файла " + file + "...")
    with open(directory + file) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            for column in row:
                total_number_of_values += 1
                if column == 0:
                    total_number_of_zero_values += 1
print("Всего значений: " + str(total_number_of_values))
print("Всего нулевых значений: " + str(total_number_of_zero_values))
print("Нулевые значения составляют " + str(
    100 - (100 * total_number_of_zero_values / total_number_of_values)) + "% от всех значений")
