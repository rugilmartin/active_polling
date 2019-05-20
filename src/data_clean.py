#For data cleaning
import csv
import numpy as np
import pandas as pd

#Sean work under here

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

def clean_census():
    in_filename = "../data/census.csv" 
    out_filename = "../data/census_clean.csv"
    headers, data = read_file(in_filename)
    data = data[data[:,1] != 'Puerto Rico']
    data[:,[1, 2]] = data[:,[2, 1]]
    write_file(out_filename, headers, data)


def combine_census_returns():
    returns_name = "../data/returns_clean.csv"
    census_name = "../data/census_clean.csv"
    returns_headers, returns_data = read_file(returns_name)
    census_headers, census_data = read_file(census_name)
    data = np.insert(census_data, range(3, 11), returns_data[:,range(3, 11), axis = 1)

    


#Raymond work under here


