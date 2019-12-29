import json
import os
import csv

rel_to_clu_dict = {}
with open("relationships.json", "r") as relationships_json:
    relationships_list = json.load(relationships_json)
    for relationship in relationships_list:
        if relationship['RelationshipDegreeNumber'] is not None and relationship['ClusterNumber'] is not None:
            rel_to_clu_dict[relationship['RelationshipDegreeNumber']] = relationship['ClusterNumber']
print(rel_to_clu_dict)

directory_old = './test_output_old/'
directory_new = './test_output/'

try:
    os.mkdir(directory_new)
except OSError as OSE:
    print(OSE)

for file in os.listdir(directory_old):
    print("Чтение файла " + file + "...")
    new_list = ''
    with open(directory_old + file, mode='r') as csvfile:
        new_list = list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC))
        i = 0
        for row in new_list:
            j = 0
            for column in row:
                new_list[i][j] = rel_to_clu_dict[column]
                j = j + 1
            i = i + 1
    with open(directory_new + file, mode='w', newline="") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(new_list)
