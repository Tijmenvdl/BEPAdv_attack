'''
This is the main executable tool for the repository of this Bachelor End Project.
'''
# Imports
from modules.data_check import data_checker

# Required datasets
datasets = [
    "Amazon_product_reviews.csv",
    "Starbucks_reviews.csv",
    "Restaurant_reviews.csv",
    "Hotel_reviews.csv"     
]

def main():
    '''
    Main executable tool
    '''
    data_checker(datasets)

if __name__ == "__main__":
    # Run main function
    main()
