import csv

def read_csv(file_path):
    reader = csv.reader(open(file_path))
    data = []
    for row in reader:
        data.append(row)
    return data[1:]