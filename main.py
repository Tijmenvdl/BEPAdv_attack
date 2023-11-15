'''
This is the main executable tool for the repository of this Bachelor End Project.
'''
# Imports
import warnings
import os

import gensim.downloader
import language_tool_python
from sentence_transformers import SentenceTransformer
import pandas as pd
from pandas.errors import SettingWithCopyWarning

from modules.data_preprocesser import preprocesser
from modules.lexicon import lexicon
from modules.manual_attack import perturb_df
from modules.success_measures import analysis_overview

warnings.simplefilter(action="ignore", category=(FutureWarning, SettingWithCopyWarning, UserWarning))

#ignore warnings
pd.set_option("mode.chained_assignment", None)

# Required datasets in dictionary with string as key and folder location for easy use
datasets = {
    "amazon": "./data/Amazon_product_reviews.csv",
    "starbucks": "./data/Starbucks_reviews.csv",
    "hotels": "./data/Hotel_reviews.csv",
    "restaurants": "./data/Restaurant_reviews.csv"
}

def main(wordsim, sentsim):
    '''
    Main executable tool
    '''
    # Loading and pre-processing datasets
    used_datasets = preprocesser(datasets, 5000) # 5000 chosen to ensure validity of p-values

    # Loading lexicons
    wordlex = lexicon()

    # Loading GloVe embeddings
    print("Downloading GloVe embeddings...")
    glove_vectors = gensim.downloader.load("glove-wiki-gigaword-100")

    # Loading grammar correction tool
    print("Downloading language tool for grammar correction tasks...")
    lang_tool = language_tool_python.LanguageTool("en-US")

    # Loading sentence similarity model
    print("Downloading sentence similarity model...")
    sent_sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Attacking the dataframes
    for file, df in used_datasets.items():

        #Computation takes long and is (fairly) deterministic so does not need to be reproduced every run
        if not os.path.isfile(rf"./data/results/attacked_{file}{int(wordsim * 100)}{int(sentsim * 100)}"):
            print(rf"Attacking {file}...")
            attacked_df = perturb_df(df,
                                     wordlex, lang_tool,
                                     glove_vectors, sent_sim_model,
                                     wordsim, sentsim)
            attacked_df.to_csv(rf"./data/results/attacked_{file}{int(wordsim * 100)}{int(sentsim * 100)}", index=False)

    # Analysing successfulness
    print("Performing successfulness analysis...")
    analysis_overview(datasets,wordsim, sentsim)

    print("Full analysis complete.")

if __name__ == "__main__":
    # Run main function
    main(wordsim=0.7, sentsim=0.8) # These may be changed to liking, but have an effect on the entire analysis
