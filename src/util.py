import numpy as np
from data_clean import read_file, write_file

def basic_data():
    headers, data = read_file('../data/full_clean_data.csv')
    y = data[:,[15,16]].astype(float)
    x = np.delete(data, [0, 1, 2, 13,14,15,16], axis = 1).astype(float)
    return x, y

