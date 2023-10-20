'''
Module that contains function to prepare 
proper directory structure and load data, if found necessary
'''

import os
import sys

def data_checker(datasets_: list):
    '''
    Inspects the current environment, and creates dictionaries if they are missing.
    Downloads datasets if they are not found to exist yet.
    Also downloads GloVe pre-trained word embeddings.
    '''
    #Create "data" and "word_embeddings" folder, if they do not exist yet
    for folder in ["./data", "./word_embeddings"]:
        if not os.path.isdir(folder):
            os.mkdir(folder)

    #Check if datasets provided in main are all found in the right location
    files_bool, missing_files = False, []
    for dataset in datasets_:
        if not os.path.isfile("./data/" + dataset):
            missing_files.append(dataset)
            files_bool = True

        #Output what files are missing
        if files_bool:
            print(rf"File(s) {missing_files} is missing. Please refer to instructions in README.md and download it...")
            sys.exit()
