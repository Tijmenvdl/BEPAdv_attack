'''
This is the main executable tool for the repository of this Bachelor End Project.
'''
# Imports
from modules.data_preprocesser import preprocesser
from modules.lexicon import lexicon

# Required datasets in dictionary with string as key and folder location for easy use
datasets = {
    "amazon": "./data/Amazon_product_reviews.csv",
    "starbucks": "./data/Starbucks_reviews.csv",
    "hotels": "./data/Hotel_reviews.csv",
    "restaurants": "./data/Restaurant_reviews.csv"
}

def main():
    '''
    Main executable tool
    '''
    # Loading and pre-processing datasets
    used_datasets = preprocesser(datasets)

    # Loading lexicons
    wordlex, wordlex_full = lexicon()

if __name__ == "__main__":
    # Run main function
    main()
