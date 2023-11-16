'''
Module contains class with functions necessary for evaluating the successfulness of attacks.
'''
import ast
import statistics
from math import log

import pandas as pd
import numpy as np
import scipy.stats as stats

class SuccessMeasures:
    '''Class contains functions for rate of found attacks over all entries, 
    as well as functions used to calculate business-related successfulness and analytical success using statistical tests.'''
    def __init__(self, df_, wordsim_, sentsim_):
        self.csv_title = df_
        self.wordsim = wordsim_
        self.sentsim = sentsim_
        self.df = pd.read_csv(rf"./data/results/{int(self.wordsim * 100)}{int(self.sentsim * 100)}attacked_{df_}").fillna("")
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
        Function that calculates base statistics, and performs statistical tests.
        Mean, std, median, and mean of log-transformed data is computed.
        Firstly, the Shapiro-Wilks test for normality is employed, upon which either the two-sample t-test
        or the Mann-Whitney U Test for equality of distribution on scores. 
        '''
        # Combining the dfs because the spectrums most be analysed together, also the unchanged ones
        df = pd.concat([self.attack_df, self.no_attack_df], ignore_index=True)

        stat_data = []
        for emotion in ["anger", "sadness", "disgust", "fear", "joy", "anticipation", "surprise", "trust"]:
            spectrum_pre, spectrum_post = df[emotion].to_list(), df[rf"{emotion}_new"].to_list()
            log_spectrum_pre = [log(x + 0.0001) for x in spectrum_pre]
            log_spectrum_post = [log(x + 0.001) for x in spectrum_post] # Smoothing applied for logarithm 

            # Calculate base stats
            mean, mean_new = statistics.fmean(spectrum_pre), statistics.fmean(spectrum_post)
            std, std_new = statistics.stdev(spectrum_pre), statistics.stdev(spectrum_post)
            median, median_new = statistics.median(spectrum_pre), statistics.median(spectrum_post)
            log_mean, log_mean_new = statistics.fmean(log_spectrum_pre), statistics.fmean(log_spectrum_post)
            log_var, log_var_new = statistics.variance(log_spectrum_pre), statistics.variance(log_spectrum_post)

            # Perform Shapiro-Wilks test for normality
            data = spectrum_pre + spectrum_post
            shapiro_stat = stats.shapiro(data)[1]

            if shapiro_stat < 0.05:

                # Mann-Whitney U Test in case of no normality
                equality_stat = stats.mannwhitneyu(spectrum_pre, spectrum_post)[1]

            else:

                # Two-sample t-test in case of normality (not expected)
                equality_stat = stats.ttest_ind(spectrum_pre, spectrum_post)[1]
            
            stat_data.append([emotion, 
                              mean, std, median, log_mean, log_var, 
                              mean_new, std_new, median_new, log_mean_new, log_var_new, 
                              shapiro_stat, equality_stat])

        return stat_data
   
def analysis_overview(dict_, wordsim, sentsim):
    '''
    Function that performs a collection of successfulness tests on all provided dfs and writes results to clearly labelled csvs.
    Parameters:
        -dict_: dictionary of active dataframes in the analysis.
        -wordsims and sentsims float values to capture the right file.
    '''

    stat_data = [] # Initialising list for df building
    for file, _ in dict_.items():

        topic = file.split("_")[0]
        analysis_item = SuccessMeasures(file, wordsim, sentsim)

        # Creating dataframe for dataframe level statistics, including business-relevant metric
        stat_data.append([topic,
                     len(analysis_item.df),
                     analysis_item.initial_success(),
                     analysis_item.business_success()[1]])
        
        # Creating dataframe for emotion-level statistics, including analytically-relevant metrics
        topic_df = pd.DataFrame(data=analysis_item.analytical_success(),
                                columns=["emotion",
                                         "mean", "std", "median", "log_mean", "log_var",
                                         "mean_new", "std_new", "median_new", "log_mean_new", "log_var_new", 
                                         "Shapiro_Wilks_pval", "Equality_test_pval"]).set_index("emotion")
        
        # Calculate effect size using Cohen's d-metric on log-transformed data
        n = len(analysis_item.df)
        topic_df["pooled_var"] = np.sqrt(((n - 1) * topic_df["log_var"] + (n - 1) * topic_df["log_var_new"]) / (n * 2 - 2))
        topic_df["Cohens_d_effect_size"] = abs(topic_df["log_mean"] - topic_df["log_mean_new"]) / topic_df["pooled_var"]
        
        # Writing topic_dfs to results folder
        topic_df = topic_df.drop(["pooled_var", "log_var", "log_var_new"], axis=1)
        topic_df.to_csv(rf"./data/results/{int(wordsim * 100)}{int(sentsim * 100)}{topic}_statistical_evaluation.csv")
    
    # Constructing dataframe out of collected successfulness measures
    analysis_df = pd.DataFrame(data=stat_data, columns=["topic",
                                                   "df_length",
                                                   "attack_rate",
                                                   "business_success"])
    analysis_df.to_csv(rf"./data/results/overall_tests_{int(wordsim * 100)}{int(sentsim * 100)}.csv")
        
    return None
    