''''
Module containing functions that do pre-processing of the lexicon
using the available resources.
'''
import pandas as pd
import numpy as np

def lexicon():
    '''
    Function loads lexicon from resource folder and performs some alterlations conforming to the research design.
    Input:
        -Nothing
    Output:
        -Wordlex_full: Full, unaltered wordlex
        -wordlex: altered wordlex
    '''

    # load wordlex
    wordlex = pd.read_csv("./lexicon/wordlex.txt", 
                          sep='\t', 
                          header=None, 
                          names=["word", "emotion", "classification"])

    # pivot table around emotions
    wordlex = wordlex.pivot_table(values="classification",
                        index="word",
                        columns="emotion",
                        aggfunc="first")

    wordlex_full = wordlex.copy()

    # drop positive and negative
    wordlex = wordlex.drop(["positive", "negative"], axis=1)

    	# Re-arrange columns, so that positive and negative emotions are adjacent
    wordlex = wordlex[["anger",
                       "disgust",
                       "fear",
                       "sadness",
                       "anticipation",
                       "joy",
                       "surprise",
                       "trust"]]

    # Overall spectrum column
    wordlex["spectrum"] = wordlex.values.tolist()

    spectrum_sums = []
    for item, _ in wordlex.iterrows():
        spectrum_sums.append(sum(wordlex.loc[item, "spectrum"]))

    # Delete words that are not associated with any emotions, 
    # these are worth just as much as unknown words to the attacker model, and should be categorised as such
    wordlex["spectrum_sum"] = spectrum_sums 
    wordlex = wordlex[wordlex["spectrum_sum"] > 0].drop(["spectrum_sum"], axis=1) # lexicon reduced by 68%

    print("Lexicon loaded...")

    return wordlex

