'''
Module that contains function to prepare 
proper directory structure and load data, if found necessary
'''

import os
import sys

import pandas as pd

from config import toggles

def preprocesser(datasets_: dict, nrows_: int): 
    '''
    Inspects the current environment, and creates folders if they are missing
    Input: 
    - datasets_ dictionary as provided in main.py with desired dataset locations. Can be adjusted in main, but not recommended
    - nrows_: int set as delimiter on how many lines are loaded
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

    print("Data folders correctly structured. Proceeding to analysis...")

    used_datasets = {}

    # preprocessing of Amazon dataset
    if toggles.data_toggles["amazon"]:
        df_amazon = pd.read_csv(datasets_["amazon"], 
                                usecols=["reviews.text"],
                                nrows=nrows_).rename(columns={"reviews.text": "text"}).fillna("")
        df_amazon = df_amazon[df_amazon["text"] != ""].sample(n=nrows_, random_state=5)
        used_datasets["Amazon_product_reviews.csv"] = df_amazon

    # preprocessing of Starbucks dataset
    if toggles.data_toggles["starbucks"]:
        df_starbucks = pd.read_csv(datasets_["starbucks"], 
                                   usecols=["Review"],
                                   nrows=nrows_).rename(columns={"Review": "text"})
        df_starbucks = df_starbucks[~(df_starbucks["text"].isin(["", "No Review Text"]))]
        used_datasets["Starbucks_reviews.csv"] = df_starbucks

    # preprocessing of hotel dataset
    if toggles.data_toggles["hotels"]:
        df_hotels = pd.read_csv(datasets_["hotels"], 
                                usecols=["Review"],
                                nrows=nrows_).rename(columns={"Review": "text"}).sample(n=nrows_, random_state=5)
        used_datasets["Hotel_reviews.csv"] = df_hotels

    # preprocessing of restaurant dataset
    if toggles.data_toggles["restaurants"]:
        df_restaurants = pd.read_csv(datasets_["restaurants"], 
                                     usecols=["Review"],
                                     nrows=nrows_).rename(columns={"Review": "text"})
        df_restaurants = df_restaurants[df_restaurants["text"].str.len() > 50] # arbitrarily chosen cut-off to include only full-sentence reviews.
        used_datasets["Restaurant_reviews.csv"] = df_restaurants

    # Decapitalisation is necessary for perturbation
    for dataset in list(used_datasets.values()):
        dataset["text"] = dataset["text"].str.lower()

    print("Dataset pre-processing good...")

    return used_datasets
