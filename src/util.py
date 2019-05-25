import numpy as np
import csv

def read_file(in_filename):
    with open(in_filename) as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [row for row in reader]
    return headers, np.array(data)

def write_file(out_filename, headers, data):
    with open(out_filename, 'w') as out_file:
        writer = csv.writer(out_file, delimiter = ',')
        writer.writerow(headers)
        writer.writerows(data)

def basic_data():
    headers, data = read_file('../data/full_clean_data.csv')
    y = data[:,[15,16]].astype(float)
    x = np.delete(data, [0, 1, 2, 13,14,15,16], axis = 1).astype(float)
    return x, y

