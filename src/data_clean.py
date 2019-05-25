#For data cleaning
import csv
import numpy as np
import pandas as pd
from util import write_file, read_file


#Sean work under here


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
    first_slice = [[i] for i in range(0,3140)]
    returns_data[first_slice, range(4,12)] = returns_data[first_slice, range(4,12)].astype(float).astype(int) 
    
    combine_headers = ['CountyID', 'State', 'County', 'TotalPop', '2004_rep', '2004_dem', '2004_dem_percent', '2008_rep', '2008_dem', '2008_dem_percent', '2012_rep', '2012_dem', '2012_dem_percent', '2016_rep', '2016_dem', '2016_dem_percent', '2016_turnout', 'male/pop', 'hispanic/pop', 'white/pop', 'black/pop', 'native/pop', 'asian/pop', 'pacific/pop', 'voting_age_citizens/pop', 'income', 'income_per_cap', 'poverty', 'child_poverty', 'professional', 'service', 'office', 'construction', 'production', 'drive', 'carpool', 'transit', 'walk', 'other_transport', 'work_at_home', 'mean_compute', 'employed/pop', 'private_work', 'public_work', 'self_employed', 'family_work', 'unemployment']


    combine_data = np.empty([len(returns_data), len(combine_headers)], dtype = object)
    for i in range(len(returns_data)):
        assert len(census_data[np.where(census_data[:,0] == returns_data[i,1])]) == 1, "County surjection"
        c_data = census_data[np.where(census_data[:,0] == returns_data[i,1])][0]
        votes = returns_data[i, range(4,12)].astype(float).astype(int)
        percent_dem = [float(votes[j])/(votes[j] + votes[j+1]) for j in range(0,8,2)]
        turnout = float(votes[6] + votes[7])/float(c_data[3])
        male_pop = float(c_data[4])/float(c_data[3])*100
        assert(turnout > 0 and turnout < 1), "In bounds"
        combine_data[i, range(4)] = census_data[i, range(4)]
        combine_data[i, (4,5,7,8,10,11,13,14)] = votes
        combine_data[i, (6,9,12,15)] = percent_dem
        combine_data[i, 16] = turnout
        combine_data[i, 17] = male_pop
        combine_data[i, range(18,24)] = c_data[range(6,12)]
        combine_data[i, 24] = float(c_data[12])/float(c_data[3])*100
        combine_data[i, (25,26)] = c_data[[13,15]]
        combine_data[i, range(27, len(combine_headers))] = c_data[range(17, len(c_data))]
        combine_data[i,41] = float(c_data[31])/float(c_data[3])*100
    write_file('../data/full_clean_data.csv', combine_headers, combine_data)

    #data = np.insert(census_data, range(3, 11), returns_data[:,range(3, 11), axis = 1)

    


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

combine_census_returns()
