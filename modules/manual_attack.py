'''
Module contains class ManualAttack for from-scratch approach on both string-level and df-level.
'''

#Imports
import random
import string
import re

import pandas as pd
import language_tool_python
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import tokenize
from sentence_transformers import util
from sklearn.metrics.pairwise import euclidean_distances

from modules.nrc_tool import nrc_affect_dict, nrc_affect_freqs, nrc_top_emotions

# sentence level
class ManualAttack:
    '''
    Class ManualAttack containing all functions needed for manual word-level substitution strategies.
    '''

    def __init__(self, text_,
                 lexicon_, lang_tool_,
                 embeddings_, sentence_model_,
                 word_sim_=0.70, sent_sim_=0.80):
        '''
        Initialising function. Takes a number of default parameters. 
        Only the word and sentence similarity scores should be changed for testing purposes.
        Further parameter:
            text_: A lowercase text message to be adversarially attacked.
        Returns nothing
        '''
        self.text = text_
        self.lexicon = lexicon_
        self.embeddings = embeddings_
        self.sentence_model = sentence_model_
        self.lang_tool = lang_tool_
        self.word_sim = word_sim_
        self.sent_sim = sent_sim_
        self.affect_freqs = nrc_affect_freqs(self.text)
        self.top_emotions = nrc_top_emotions(self.text)

    # ---- language check ----
       
    def lang_check(self, new_text):
        '''
        Functions performs language/grammar check on two strings.
        Will fix in second string only the fixes that are found on top of the ones found in first string.
        Parameters:
            -new_text: Second string with one word-level substitution compared to first
        Returns:
            -str2_fixed: Grammar-fixed second string, using only the fixes found on top of the first string's ones.
        '''

        # Find language errors in both string
        matches_1, matches_2 = self.lang_tool.check(self.text), self.lang_tool.check(new_text)

        # Look for that one fix
        found_fixes = []
        while not found_fixes:
            try:
                # Find the inserted error
                # Only if there is a higher amount of errors in the new string, a correction must take place.
                if len(matches_1) < len(matches_2):
                    for i, match in enumerate(matches_2):
                        if match.message != matches_1[i].message:
                            found_fixes.append(match)
                # In any other case or if found_fixes stays empty, the while-loop is aborted
                break
            # In case the previous loop leads to an indexerror, we have arrived at the final message
            except IndexError:
                found_fixes.append(matches_2[-1])

        new_text_fixed = language_tool_python.utils.correct(new_text, found_fixes)
        return new_text_fixed
    
    # ---- sentence similarity ----

    def sentence_similarity(self, sent_list):
        '''
        Function to evaluate sentence similarity between original text message and perturbed sentence.
        Parameters:
            -sent_list: Two-item list containing, firstly, the original text and secondly, the perturbed sentence
        Returns
            -sent_simil: Boolean, True if sentence similarity is above set threshold in __init__.
        '''
        sent_simil = False
        sent_embedding_1 = self.sentence_model.encode(sent_list[0], convert_to_tensor=True)
        sent_embedding_2 = self.sentence_model.encode(sent_list[1], convert_to_tensor=True)
        if util.pytorch_cos_sim(sent_embedding_1, sent_embedding_2).item() >= self.sent_sim:
            sent_simil = True
        return sent_simil
    
    # ---- emotionless pipeline

    def non_emotional_replacement(self, word):
        '''
        Function that looks for perturbation candidates if no emotional word was found in the text.
        Parameter:
            -word: A word isolated from the text message.
        Returns:
            -most_similar_dict: Dictionary with similarity scores containing similar words that do possess some emotion
        '''
        most_similar = self.embeddings.most_similar(word, topn=50) # Find most similar words, number 50 is arbitrarily chosen
        most_similar_dict = list({key: val for key, val in most_similar if key in self.lexicon.index}.keys()) # perturbation candidate must have some emotion
        return most_similar_dict
    
    def non_emotional_pipeline(self):
        '''
        Function that performs full word-substitution algorithm in case of emotionlessness in original text.
        Returns:
            -new_text: perturbed sentence that passes all checks
            OR a statement of unsuccessfulness
        '''

        #target words become cleaned tokens of sentence
        clean_sentence = remove_stopwords(self.text)
        target_words_prio = list(tokenize(clean_sentence.translate(str.maketrans("", "", string.punctuation))))
        random.shuffle(target_words_prio)

        for word in target_words_prio:
            if word in self.embeddings: # target word must be in GloVe vocab
                for candidate in self.non_emotional_replacement(word):
                    new_text = self.lang_check(re.sub(word, candidate, self.text))

                    # The string must comply to three requirements
                    # 1) The sentence must be semantically similar enough
                    # 2) The language corrector may not grammar-correct the perturbed sentence back to itself,
                    # 3) The language corrector may not grammar-correct the perturbed sentence to a version,
                    # that changes the sentiment of the perturbation candidate (conjugations that are not fully encapsulated in wordlex).
                    if new_text != self.text and self.sentence_similarity([self.text, new_text]) and nrc_affect_freqs(new_text) != self.affect_freqs:
                        return new_text # return attack if found similar enough
        return "No adversarial attack found." # statement of unsuccessfulness
    
    # ---- emotional pipeline ----
    
    def emotional_replacement(self, word):
        '''
        Function that looks for perturbation candidates if emotional words are found in the text.
        Parameter:
            -word: A word isolated from the text message.
        Returns:
            -most_similar_dict: Perturbation candidates for a word ranked on damage and similarity.
        '''
        # create dataframe with similarity scores for similar words
        most_similar = pd.DataFrame(self.embeddings.most_similar(word, topn=50),
                                    columns=["word", "sim_score"]) # Find most similar words, number 50 is arbitrarily chosen
        most_similar_cutoff = most_similar[most_similar["sim_score"] >= self.word_sim] # Not strictly necessary, but speeds up computation (restricts search space)

        # create column with spectrum (existing or empty) to compare with input word
        most_similar_cutoff["spectrum"] = most_similar_cutoff["word"].apply(lambda x: self.lexicon.loc[x, "spectrum"] if x in self.lexicon.index else [0] * 8)
        
        # compute Euclidean distance between original word spectrum vector and replacement candidates
        current_spectrum = self.lexicon.loc[word, "spectrum"]
        dists = []
        for item in most_similar_cutoff["spectrum"]:
            dists.append(euclidean_distances([item], [current_spectrum]))
        most_similar_cutoff["dist"] = [item.item() for item in dists]

        # sort candidates by greatest Euclidean spectrum distance and similarity scores
        return most_similar_cutoff.sort_values(by=["sim_score", "dist"], ascending=False)["word"].values.tolist()
    
    def emotional_pipeline(self, target_words):
        '''
        Function that performs full word-substitution algorithm in case of emotionality in original text.
        Parameters:
            Target_words: List of words that contain emotion and can thus be substituted.
        Returns:
            -new_text: perturbed sentence that passes all checks
            OR a statement of unsuccessfulness
        '''
        for word in target_words:
            if word in self.embeddings: # target word must be in GloVe vocab, correct for small discrepancies in tool/lexicon
                for candidate in self.emotional_replacement(word):
                    new_text = self.lang_check(re.sub(word, candidate, self.text)) # language check

                    #see comments above for conditions explanation
                    if new_text != self.text and self.sentence_similarity([self.text, new_text]) and nrc_affect_freqs(new_text) != self.affect_freqs:
                        return new_text # return attack if found similar enough
        return "No adversarial attack found." # statement of unsuccessfulness
    
    # ---- full pipeline ----

    def full_pipeline(self):
        '''
        Full_pipeline function serves for nothing more than to combine all functions into one function to call on dataframe instances.
        Returns:
            Perturbed text.
        '''
        #Take prioritised list of target words from nrc functions and apriori affect_freqs and top_emotions
        target_words_prio = list(nrc_affect_dict(self.text).keys())
        target_words_prio = [word for word in target_words_prio if word in self.lexicon.index]

        # If sentece is currently emotionless, we will try to attack some emotion into them.
        if not target_words_prio:
            return self.non_emotional_pipeline()
        
        # Otherwise attack will be attacked out of them, or changed over the emotions spectrum
        return self.emotional_pipeline(target_words_prio)

# df-level
def perturb_df(dataset_, lexicon_, lang_tool_, embeddings_, sent_sim_model_):
    '''
    Performs dataset-level perturbations using ManualAttack.
    Parameters:
        dataset_: Pandas DataFrame, conforming to the formatting of data_preprocesser.py
    Output:
        new_df: Perturbed dataset
    '''
    # Create affect frequencies and top emotion as df columns
    new_df = dataset_.copy()
    new_df["freqs"] = new_df["text"].apply(lambda x: nrc_affect_freqs(x))
    new_df = pd.concat([new_df.drop(["freqs"], axis=1),
                        new_df["freqs"].apply(pd.Series)],
                        axis=1)
    new_df["top_emotion"] = new_df["text"].apply(lambda x: nrc_top_emotions(x))

    # Create df column with perturbed text using ManualAttack
    new_df["text_new"] = new_df["text"].apply(lambda x: ManualAttack(x,
                                                                     lexicon_, lang_tool_,
                                                                     embeddings_, sent_sim_model_).full_pipeline())

    # Find their affect frequencies
    empty_dict = {"anger": "",
                  "disgust": "",
                  "fear": "",
                  "sadness": "",
                  "anticipation": "",
                  "joy": "",
                  "surprise": "",
                  "trust": ""}
    new_df["freqs_new"] = new_df["text_new"].apply(lambda x: nrc_affect_freqs(x) if x != "No adversarial attack found." else empty_dict)
    new_df = pd.concat([new_df.drop(["freqs_new"], axis=1),
                        new_df["freqs_new"].apply(pd.Series).add_suffix("_new")],
                        axis=1)
    new_df["top_emotions_new"] = new_df["text_new"].apply(lambda x: nrc_top_emotions(x) if x != "No adversarial attack found." else "")
    

    return new_df
