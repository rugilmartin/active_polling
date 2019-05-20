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

def clean_returns(returns_filepath, census_filepath):
    full = pd.read_csv(returns_filepath, header=0)
    county_id = pd.read_csv(census_filepath,
                            header=0,
                            usecols=['CensusId', 'State', 'County'])

    for year in [2004, 2008, 2012, 2016]:
        clean = pd.DataFrame({"census_id": county_id.CensusId,
                              "county": county_id.County,
                              "state": county_id.State})
        clean = clean[clean.state != 'Puerto Rico']
        clean["dem" + str(year)] = np.zeros(clean.shape[0])
        clean["rep" + str(year)] = np.zeros(clean.shape[0])
        for i in range(clean.shape[0]):
            c_id = clean.census_id[i]
            demrow = full.loc[(full['FIPS'] == c_id) & (full['year'] == year) & (full['party'] == "democrat")]
            reprow = full.loc[(full['FIPS'] == c_id) & (full['year'] == year) & (full['party'] == "republican")]
            if demrow.empty:
                print(year, " ", c_id, " dem")
            else:
                clean["dem" + str(year)][i] = demrow.candidatevotes[demrow.index[0]]
            if reprow.empty:
                print(year, " ", c_id, " rep")
            else:
                clean["rep" + str(year)][i] = reprow.candidatevotes[reprow.index[0]]
        print("Completed ", year)
        clean.to_csv("data/returns_5_20.csv", mode="a+")

