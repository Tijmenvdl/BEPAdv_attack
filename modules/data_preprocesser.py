'''
Module that contains function to prepare 
proper directory structure and load data, if found necessary
'''

import os
import sys

import pandas as pd

def preprocesser(datasets_: dict): 
    '''
    Inspects the current environment, and creates folders if they are missing
    Input: 
    - datasets_ dictionary as provided in main.py with desired dataset locations. Can be adjusted in main, but not recommended
    '''
    #Create "data" folder, if it does not exist yet
    if not os.path.isdir("./data"):
        os.mkdir("./data")

    #Check if datasets provided in main are all found in the right location
    files_bool, missing_files = False, []
    for dataset in list(datasets_.values()):
        if not os.path.isfile(dataset):
            missing_files.append(dataset)
            files_bool = True

        #Output what files are missing
    if files_bool:
        print(rf"File(s) {missing_files} is missing. Please refer to instructions in README.md and download (AND RENAME) accordingly...")
        sys.exit()

    print("Datasets correctly structured. Proceeding to analysis...")

    # preprocessing of Amazon dataset
    df_amazon = pd.read_csv(datasets_["amazon"], usecols=["reviews.text"]).rename(columns={"reviews.text": "text"}).fillna("")
    df_amazon = df_amazon[df_amazon["text"] != ""]

    # preprocessing of Starbucks dataset
    df_starbucks = pd.read_csv(datasets_["starbucks"], usecols=["Review"]).rename(columns={"Review": "text"})
    df_starbucks = df_starbucks[~(df_starbucks["text"].isin(["", "No Review Text"]))]

    # preprocessing of hotel dataset
    df_hotels = pd.read_csv(datasets_["hotels"], usecols=["Review"]).rename(columns={"Review": "text"})

    # preprocessing of restaurant dataset
    df_restaurants = pd.read_csv(datasets_["restaurants"], usecols=["Review"]).rename(columns={"Review": "text"})
    df_restaurants = df_restaurants[df_restaurants["text"].str.len() > 50] # arbitrarily chosen cut-off to include only full-sentence reviews.

    return df_amazon, df_starbucks, df_hotels, df_restaurants
