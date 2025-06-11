# The goal of this script is to clean the scraped dataframes so that a primary key can be used to merge based on player name and position. In cases where there are same name/positions, I address it by adding drafted year. 


import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
# Suppress all warnings
warnings.filterwarnings('ignore')

import string



# Make a feature to flag if a player came from a power five school
power_five_schools = ['Alabama', 'Georgia', 'LSU', 'Michigan', 'Florida', # power five schools plus Notre Dame and Uconn
                      'Notre Dame','Clemson', 'Oklahoma', 'Oregon', 'Auburn', 
                      'Texas', 'Washington', 'USC','Ohio State', 'Mississippi', 
                      'Iowa', 'Texas A&M', 'Ohio St.','Arkansas', 'Utah', 'South Carolina',
                      'Wisconsin', 'Kentucky', 'UCLA','Stanford', 'TCU', 'Louisville', 
                      'Missouri', 'Pittsburgh', 'Miami (FL)','Florida State', 
                      'North Carolina', 'Tennessee', 'Virginia Tech', 'Penn St.', 
                      'Maryland', 'Nebraska', 'Baylor', 'Cincinnati', 'West Virginia', 
                      'Minnesota', 'Penn State', 'Indiana', 'Mississippi State', 'Miami', 
                      'Michigan State', 'Central Florida', 'Boston Col.', 'Illinois', 
                      'Texas Tech', 'Northwestern', 'Syracuse', 'North Carolina State',
                      'Florida St.', 'Colorado', 'California', 'Houston','Kansas St.', 
                      'BYU','Purdue', 'Duke', 'Arizona State', 'Kansas', 'Virginia',
                      'Wake Forest', 'Oklahoma St.', 'Rutgers', 'Iowa St.', 'Arizona',
                      'Georgia Tech', 'Washington State', 'Oklahoma State',  'Vanderbilt',
                      'Arizona St.',  'Mississippi St.','SMU', 'Michigan St.', 
                      'Kansas State','North Carolina St.', 'Washington St.', 'Boston College'] 


# rollup position from multiple data sources
position_rollup = {
    'WR':['WR', 'CB/WR', 'wide-receiver'], 
    'CB':['CB', 'cornerback'],
    'RB':['RB', 'FB', 'HB', 'running-back', 'fullback'],
    'OT':['OT', 'LT', 'RT', 'left-tackle', 'right-tackle'],
    'LB':['LB', 'OLB', 'ILB', 'LOLB', 'ROLB', 'MLB', 'linebacker'],
    'DL':['DL','DT', 'interior-defensive-line'],
    'TE':['TE', 'tight-end'],
    'DB':['DB', 'S', 'LS', 'SAF', 'FS', 'SS', 'safety'],
    'QB':['QB', 'quarterback'],
    'OL':['OL', 'OG', 'C', 'G', 'LG', 'RG', 'left-guard', 'center', 'right-guard'],
    'K':['K', 'kicker'],
    'P':['P', 'punter'],
    'LS':['long-snapper'],
    'EDGE':['DE', 'EDGE', 'RE', 'LE', 'edge-rusher']
}

def replace_school_names(data):
    school_name_replacement_dict = {}
    
    for school in [i for i in data.School.unique() if 'St.' in i]:
        school_name_replacement_dict[school] = school.replace('St.', 'State')
    
    for school in [i for i in data.School.unique() if 'Col.' in i]:
        school_name_replacement_dict[school] = school.replace('Col.', 'College')

    return(school_name_replacement_dict)

def replace_position(data, position_col):
    """position_col is the name of the position column for the dataset"""
    # Rollup position groups 
    pos_replace_dict = {}
    for i in position_rollup.keys():
        for j in position_rollup[i]:
            pos_replace_dict[j] = i
        
    data['rolled_up_pos'] = data[position_col].replace(pos_replace_dict)
    return(data)


# Load data to clean 
player_salaries = pd.read_csv('../data/player_salaries.csv').rename(columns = {'Team':'Team25'})
madden_ratings = pd.read_csv('../data/madden_ratings.csv').rename(columns = {'Team':'Team24'})
replace_position(player_salaries, 'position')
replace_position(madden_ratings, 'Position')
madden_ratings.loc[:,'Team24'] = madden_ratings.apply(lambda row: row['Team24'].split(' ')[-1], axis=1)

combine_data = pd.read_csv('../data/full_combine_data.csv')




def read_in_combine_data(year_range = (2015, 2025)):
    """
    Loads in the data and performs some preprocessing
    
    Inputs:
        year_range (tuple) - first and last year with available combine data

    Outputs:
        full_data (DF) - combine data with some added engineered features
    
    
    """

    # Create an empty data frame with the columns of interest
    full_data = pd.DataFrame(columns = ['Player', 'Pos', 'School', 'Ht', 'Wt', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'drafted_team', 'drafted_round', 'drafted_pick', 'year'])


    # Iteratively read in the data for each year 
    for year in range(year_range[0], year_range[1]+1):
        # load csv
        data = pd.read_csv(f'../data/combine_data_{year}.txt')
        
        # split the Drafted (tm/rnd/yr) column into the drafted team, round, and pick columns
        data['drafted_team'] = data.apply(lambda row: row['Drafted (tm/rnd/yr)'] if str(row['Drafted (tm/rnd/yr)']) == 'nan' else row['Drafted (tm/rnd/yr)'].split('/')[0], axis=1)
        data['drafted_round'] = data.apply(lambda row: row['Drafted (tm/rnd/yr)'] if str(row['Drafted (tm/rnd/yr)']) == 'nan' else row['Drafted (tm/rnd/yr)'].split('/')[1][:-3], axis=1).astype(float)
        data['drafted_pick'] = data.apply(lambda row: row['Drafted (tm/rnd/yr)'] if str(row['Drafted (tm/rnd/yr)']) == 'nan' else row['Drafted (tm/rnd/yr)'].split('/')[2][:-8], axis=1).astype(float)

        # add a drafted flag
        data['drafted'] = data.apply(lambda row: 1 if row['drafted_round'] in [1,2,3,4,5,6,7] else 0, axis =1)

        # convert height to inches
        data['Ht'] = data.apply(lambda row:row['Ht'] if str(row['Ht']) == 'nan' else 12*int(row['Ht'].split('-')[0])+int(row['Ht'].split('-')[1]), axis=1)

        # add year of combine
        data['year'] = [year]*len(data)

        # add a binary flag for if a player came from a power 5 school (using 2025 definitions)
        data['power5'] = data.apply(lambda row: 1 if row['School'] in power_five_schools else 0, axis=1)

        # Rollup position groups 
        pos_replace_dict = {}
        for i in position_rollup.keys():
            for j in position_rollup[i]:
                pos_replace_dict[j] = i
        
        data['rolled_up_pos'] = data.Pos.replace(pos_replace_dict)

        # drop irrelevant columns
        data = data.drop(columns = ['Drafted (tm/rnd/yr)', 'Player-additional', 'College'])

        # add this year's data to the full dataframe
        full_data = pd.concat([full_data, data])

    # update school names for consistency
    full_data.School = full_data.School.replace(replace_school_names(full_data))

    # read in coach salary data
    coach_salaries = pd.read_csv('../data/coach_salaries.csv')

    # add coaches salary to data
    full_data = full_data.merge(coach_salaries, left_on = 'School', right_on = 'school', how='left')

    full_data['log_coach_salary'] = round(np.log10(full_data['salary']), 2)

    full_data = full_data.drop(columns = ['school'])

    full_data['draft_night'] = pd.cut(full_data['drafted_round'], bins=[0,1.5,3.5,7.5], labels=[1,2,3]).astype(float).fillna(4)


    return(full_data)






def clean_string(s):
    translator = str.maketrans('', '', string.punctuation)
    s = s.translate(translator).lower().replace(' ', '_')
    return s


madden_ratings['player_id'] = madden_ratings.apply(lambda row: clean_string(row['Player'])+'_'+row['rolled_up_pos'], axis=1)
player_salaries['player_id'] = player_salaries.apply(lambda row: clean_string(row['Player'])+'_'+row['rolled_up_pos'], axis=1)
combine_data['player_id'] = combine_data.apply(lambda row: clean_string(row['Player'])+'_'+row['rolled_up_pos'], axis=1)








# need to replace the player_ids for those below:

combine_data_id_replacement_dict = {
    528:'connor_mcgovern_OL_16', 
    1524:'connor_mcgovern_OL_19', 
    1492:'isaiah_johnson_CB_19', 
    3236:'isaiah_johnson_CB_24', 
    2303:'jarrett_patterson_OL_21', 
    2997:'jarrett_patterson_OL_23', 
    237:'jordan_phillips_DL_15', 
    3651:'jordan_phillips_DL_25', 
    1959:'michael_turk_P_20', 
    3064:'michael_turk_P_23'
}

player_salaries_id_replacement_dict = {
    1572:'jordan_phillips_DL_15', 
    1584:'jordan_phillips_DL_25', 
    2595:'jaylon_jones_CB_UD', 
    2735:'jaylon_jones_CB_23'
}

madden_ratings_id_replacement_dict = {
    712:'jaylon_jones_CB_23', 
    1195:'jaylon_jones_CB_UD'
}



for idx in combine_data_id_replacement_dict.keys():
    combine_data.iloc[idx,-1] = combine_data_id_replacement_dict[idx]

for idx in player_salaries_id_replacement_dict.keys():
    player_salaries.iloc[idx,-1] = player_salaries_id_replacement_dict[idx]

for idx in madden_ratings_id_replacement_dict.keys():
    madden_ratings.iloc[idx,-1] = madden_ratings_id_replacement_dict[idx]





#combine_data.to_csv('../data/full_combine_data.csv', index=False)
#madden_ratings.to_csv('../data/full_madden_ratings.csv', index=False)
#player_salaries.to_csv('../data/full_player_salaries.csv', index=False)

