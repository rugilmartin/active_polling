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

    clean = pd.DataFrame({"census_id": county_id.CensusId,
                          "county": county_id.County,
                          "state": county_id.State})
    # Get rid of Puerto Rico. No electoral votes.
    clean = clean[clean.state != 'Puerto Rico']

    for year in [2004, 2008, 2012, 2016]:
        clean["dem" + str(year)] = np.zeros(clean.shape[0])
        clean["rep" + str(year)] = np.zeros(clean.shape[0])
        for i in range(clean.shape[0]):
            c_id = clean.census_id[i]
            demrow = full.loc[(full['FIPS'] == c_id) & (full['year'] == year) & (full['party'] == "democrat")]
            reprow = full.loc[(full['FIPS'] == c_id) & (full['year'] == year) & (full['party'] == "republican")]
            if demrow.empty or reprow.empty:
                clean_vals =  odd_case(full, year, c_id, clean)
                clean["dem" + str(year)][i] = clean_vals[0]
                clean["rep" + str(year)][i] = clean_vals[1]
            else:
                clean["dem" + str(year)][i] = demrow.candidatevotes[demrow.index[0]]
                clean["rep" + str(year)][i] = reprow.candidatevotes[reprow.index[0]]
        print("Completed ", year)

    # Get rid of Kalawao County in Hawaii. Few residents, no votes.
    clean = clean[clean.census_id != 15005]

    clean.to_csv("data/returns_5_20.csv")



def odd_case(full, year, c_id, clean):
    id_row = clean.loc[(clean['census_id'] == c_id)]
    state = id_row.state[id_row.index[0]]
    county = id_row.county[id_row.index[0]]
    # Handle Kalawao County, Hawaii just in case.
    if c_id == 15005:
        return [0, 0]
    # Oglala Lakota County, South Dakota has wrong id number.
    elif c_id == 46102:
        c_id = 46113
        demrow = full.loc[(full['FIPS'] == c_id) & (full['year'] == year) & (full['party'] == "democrat")]
        reprow = full.loc[(full['FIPS'] == c_id) & (full['year'] == year) & (full['party'] == "republican")]
        dem = demrow.candidatevotes[demrow.index[0]]
        rep = reprow.candidatevotes[reprow.index[0]]
        return [dem, rep]
    # Handle Alaska separately.
    elif state == "Alaska":
        ak_data = pd.read_csv("data/alaska_returns.csv", header=0)
        county_row = ak_data.loc[(ak_data['ID'] == c_id)]
        dem = county_row[str(year) + "_dem"][county_row.index[0]]
        rep = county_row[str(year) + "_rep"][county_row.index[0]]
        total = county_row[str(year) + "_tot"][county_row.index[0]]
        return [dem*total/100, rep*total/100]
    else:
        print(state, county, c_id)
        return [0, 0]
