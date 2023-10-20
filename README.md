# BEPAdv_attack
Welcome to the repository for my Bachelor End Project: "Adversarial Attacks on Dictionary-Based Measurement Tools". Following the instructions in this file, should ensure smooth running of the executable tool and all its modules.

## Purpose
This repository contains all the tools that are used for my analysis. The project involves evaluating the vulnerability to adversarial attacks of the emotions classifier/quantifier based on the NRC Emotions Lexicon. I test the completeness of the lexicon by various attack methods, some from scratch, some using state-of-the-art adversarial attacker platforms. 

## Requirements
This project was written in Python version 3.11. To prevent any problems with running the code, please install this version [here](https://www.python.org/downloads/). 

If you do not have (the correct version of) <em>pip</em> installed, this can be done by running the following code in your Command Prompt:
> `python -m pip install --upgrade pip` <br>

To ensure the proper versions of all packages are installed, please run the <em>requirements.txt</em> file through your pip installer, by - once again in command prompt - running the following:
> `python -m pip install -r requirements.txt`

## Downloading datasets
When cloning this repository from GitHub, a folder with data and word embeddings is not included by default. This is done to reduce the storage size of the project on Git (hence the reason for the inclusion of the lexicon files, which are not large files), and will ask users to download the data and set the right folders themselves. The module `data_check.py` provides some help in doing this. It will install a folder `./data` and `./word_embeddings` at the right location within the main folder. 
For complexity reasons, I do not supply code that automatically downloads missing datasets. However, below one can find a rundown of what files are needed to download and extract. The files must named as put in the folder and named as specified.
<ul>
<li> data:
<ul>
<li> [Amazon_prduct_reviews.csv](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)</li>
<li> [Hotel_reviews.csv](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews/)</li>
<li> [Restaurant_reviews.csv](https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews)</li>
<li> [Starbucks_reviews.csv](https://www.kaggle.com/datasets/harshalhonde/starbucks-reviews-dataset)</li>
</ul>
<li> word_embeddings:
<ul>
<li> [GloVe word embeddings](https://nlp.stanford.edu/data/glove.6B.zip) (Download all files and leave names as is) </li>
</ul>
</ul>

## Executing tool
The file `main.py` contains calls on all relevant modules. As such, the only thing a user needs to do to reproduce results, is run this file in an IDE or the command line.
