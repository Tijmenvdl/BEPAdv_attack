'''
Module contains class with functions necessary for evaluating the successfulness of attacks.
'''
import ast

import pandas as pd
import numpy as np
import scipy.stats as stats

class SuccessMeasures:
    '''Class contains functions for rate of found attacks over all entries, 
    as well as functions used to calculate business-related successfulness and analytical success using statistical tests.'''
    def __init__(self, df_):
        self.csv_title = df_
        self.df = pd.read_csv(rf"./data/attacked_{df_}").fillna("")
        self.attack_df = self.df[self.df["text_new"] != "No adversarial attack found."]
        no_attack_df = self.df[self.df["text_new"] == "No adversarial attack found."]

        # set unperturbed sentence scores to that of the original sentence for statistical testing purposes
        for emotion in ["anger", "sadness", "disgust", "fear", "joy", "anticipation", "surprise", "trust"]:
            no_attack_df[rf"{emotion}_new"] = no_attack_df[emotion]
        self.no_attack_df = no_attack_df

    def initial_success(self):
        '''
        Function computes the initial fail rate of the attacker model in finding any perturbation to the provided texts.
        Returns:
            -Attack succesfulness
        '''
        nr_failed_attacks = self.df["text_new"].str.count("No adversarial attack found.").sum()
        fail_rate = round(nr_failed_attacks / len(self.df), 3)
        return 1 - fail_rate #successfulness rate
    
    def emotion_ratio(self, emo_lst):
        '''
        Small function used in the one below. Computes ratio of negative and positive emotions in the list of dominant ones.
        Parameter:
            emo_lst: List of initially dominant emotions.
        Returns:
            neg_count, pos_count: two float values with respective ratioes. 
        '''
        emo_lst = ast.literal_eval(emo_lst) #fix for read_csv evaluating the list as string
        neg_emotions = ["anger", "sadness", "disgust", "fear"]
        pos_emotions = ["joy", "anticipation", "surprise", "trust"]
        neg_count = len(list(set(emo_lst) & set(neg_emotions)))/len(emo_lst)
        pos_count = len(list(set(emo_lst) & set(pos_emotions)))/len(emo_lst)
        return neg_count, pos_count
        
    
    def business_success(self):
        '''
        Function that evaluates the successfulness on a business-level perspective. 
        This means a 50% change in either the positivity or negativitiy ratio in the list of dominant emotions.
        Returns:
            -full_df: Complete df with successfulness indicator
            -score: Successfulness score.
        '''
        df = self.attack_df.copy()
        
        # Rate for prevalence in initial dominant emotion lists
        df["neg_intersect"] = df["top_emotion"].apply(lambda x: self.emotion_ratio(x)[0])
        df["pos_intersect"] = df["top_emotion"].apply(lambda x: self.emotion_ratio(x)[1])
        
        # Rate for prevalence in new dominant emotion lists
        df["new_neg_intersect"] = df["top_emotions_new"].apply(lambda x: float(self.emotion_ratio(x)[0]) if x != "" else x)
        df["new_pos_intersect"] = df["top_emotions_new"].apply(lambda x: float(self.emotion_ratio(x)[1]) if x != "" else x)

        # Differences in prevalence rates
        df["neg_diff"] = abs(df["new_neg_intersect"] - df["neg_intersect"])
        df["pos_diff"] = abs(df["new_pos_intersect"] - df["pos_intersect"])

        # Defining the business-perspective measure
        df["business_measure"] = np.where(df["neg_diff"] >= 0.5,
                                               "Successful",
                                               np.where(df["pos_diff"] >= 0.5,
                                                        "Successful",
                                                        "Unsuccessful"))
        
        # Concatenating dataframes 
        full_df = pd.concat([df, self.no_attack_df], ignore_index=True)
        full_df["business_measure"] = full_df["business_measure"].fillna("Unsuccessful") # Setting unchanged sentences to unsuccessful by default
        
        # Returning the df as well as the individual rate
        return full_df, full_df["business_measure"].str.count("Successful").sum() / len(full_df)
    
    def analytical_success(self):
        '''
        Function that performs statistical tests to evaluate analytical test.
        Firstly, the Shapiro-Wilks test for normality is employed, upon which either the t-test for dependent samples
        or the Wilcoxon Signed-Rank Test for equality of distribution on scores. 
        '''
        # Combining the dfs because the spectrums most be analysed together, also the unchanged ones
        df = pd.concat([self.attack_df, self.no_attack_df], ignore_index=True)

        evals = {}
        for emotion in ["anger", "sadness", "disgust", "fear", "joy", "anticipation", "surprise", "trust"]:
            spectrum_pre, spectrum_post = df[emotion].to_list(), df[rf"{emotion}_new"].to_list()
            data = spectrum_pre + spectrum_post

            # Perform Shapiro-Wilks test for normality
            if stats.shapiro(data)[1] < 0.05:

                # Wilcoxon Signed-Rank Test in case of no normality
                wilcoxon_stat = stats.wilcoxon(spectrum_pre, spectrum_post)[1]
                evals[emotion] = {rf"{emotion}_normality p-value": stats.shapiro(data)[1],
                                  rf"{emotion}_equality_p-value": wilcoxon_stat}
            else:

                # T-test for dependent samples in case of normality (not expected)
                ttest_stat = stats.ttest_rel(spectrum_pre, spectrum_post)[1]
                evals[emotion] = {rf"{emotion}_normality p-value": stats.shapiro(data)[1],
                                 rf"{emotion}_equality_p-value": ttest_stat}

        return evals
    
def analysis_overview(dict_):
    '''
    Function that performs a collection of successfulness tests on all provided dfs
    Parameters:
        -dict_: dictionary of active dataframes in the analysis.
    Returns:
        -analysis_df: df containing collection of successfulness
    '''

    data = [] # Initialising list for df building
    for topic, file in dict_.items():

        # Correctly formatting filename
        filename = file.replace("./data/", "")
        analysis_item = SuccessMeasures(filename)
        data.append([topic, 
                     len(analysis_item.df), 
                     analysis_item.initial_success(),
                     analysis_item.business_success()[1],
                     analysis_item.analytical_success()])
    
    # Constructing dataframe out of collected successfulness measures
    analysis_df = pd.DataFrame(data=data, columns=["topic",
                                                   "df_length",
                                                   "attack_rate",
                                                   "business_success",
                                                   "statistical_success"])
    
    # Exploding dictionaries to get columns with p-values for both normality and equality tests
    analysis_df = pd.concat([analysis_df.drop("statistical_success", axis=1),
                             analysis_df["statistical_success"].apply(pd.Series)],
                             axis=1)
    
    for emotion in ["anger", "sadness", "disgust", "fear", "joy", "anticipation", "surprise", "trust"]:
        analysis_df = pd.concat([analysis_df.drop(emotion, axis=1),
                                 analysis_df[emotion].apply(pd.Series)],
                                 axis=1)
        
    return analysis_df
    