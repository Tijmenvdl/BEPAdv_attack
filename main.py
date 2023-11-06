'''
This is the main executable tool for the repository of this Bachelor End Project.
'''
# Imports
import gensim.downloader
import language_tool_python
from sentence_transformers import SentenceTransformer

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

    # Loading GloVe embeddings
    print("Downloading GloVe embeddings...")
    glove_vectors = gensim.downloader.load("glove-wiki-gigaword-100")

    # Loading grammar correction tool
    lang_tool = language_tool_python.LanguageTool("en-US")

    # Loading sentence similarity model
    print("Downloading sentence transformer...")
    sent_sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Analysis complete.")

if __name__ == "__main__":
    # Run main function
    main()
