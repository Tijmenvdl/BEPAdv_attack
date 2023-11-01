'''
This is the main executable tool for the repository of this Bachelor End Project.
'''
# Imports
from modules.data_check import data_checker

# Required datasets in dictionary with string as key and folder location for easy use
datasets = {
    "amazon": "./data/Amazon_product_reviews.csv",
    "starbucks": "./data/Starbucksreviews.csv",
    "hotels": "./data/Hotel_reviews.csv",
    "restaurants": "/.data/Restaurant_reviews.csv"
}

def main():
    '''
    Main executable tool
    '''
    data_checker(datasets)

if __name__ == "__main__":
    # Run main function
    main()
