'''
Module that contains all functions relevant for analysing data w.r.t. the NRC Emotions quantifier
'''

from collections import Counter

from nrclex import NRCLex

def nrc_affect_dict(text_: str):
    '''
    Function that returns target words for an adversarial attack
    Input:
        -text: text message that is analysed by NRCLex
    Output:
        -sorted_affect_dict: dictionary with word affects ordered on descending attack successfulness likelihood
    '''

    #Establish affect dictionary
    affect_dict = NRCLex(text_).affect_dict

    # remove elemantary sentiments from affect dictionary
    for sentiment in ["positive", "negative"]:
        affect_dict = {key: [val for val in value if val != sentiment] for key, value in affect_dict.items()}

    # remove keys that are now empty (only had elementary sentiment annotation)
    affect_dict = {key: value for key, value in affect_dict.items() if value != []}

    sorted_affect_dict = dict(sorted(affect_dict.items(),
                                     key=lambda item:len(item[1]),
                                     reverse=True))
    
    return sorted_affect_dict

def nrc_affect_freqs(text_: str):
    '''
    Function that computes affect_freqs for the messages. 
    Computation is copied from Git resources, to circumvent elementary sentiments, using the output of nrc_affect_dict.
    Input:
        -text: text message that is analysed by NRCLex
    Output:
        -affect_freqs: the emotional quantification scores on the eight prototypical emotions.
    '''

    # Create affect_list
    emotions_sublists = list(nrc_affect_dict(text_).values())
    affect_list = [emotion for sublist in emotions_sublists for emotion in sublist]

    # Create affect_freqs
    affect_percentages = {"anger": 0.0,
                          "disgust": 0.0,
                          "fear": 0.0,
                          "sadness": 0.0,
                          "anticipation": 0.0,
                          "joy": 0.0,
                          "surprise": 0.0,
                          "trust": 0.0}
    
    affect_freqs = Counter()
    for word in affect_list:
        affect_freqs[word] += 1

    sum_values = sum(affect_freqs.values())

    for key, value in affect_freqs.items():
        affect_percentages.update({key: round(value / sum_values, 2)})
        
    return affect_percentages

def nrc_top_emotions(text_: str):
    '''
    Function that return most dominant emotion(s) in a piece of text on the basis of affect_percentages form nrc_affect_freqs
    Input:
        text_: text message that is analysed by NRCLex.
    Output:
        top_emotions: list containing most dominant emotion(s)
    '''

    affect_freqs = nrc_affect_freqs(text_)
    max_keys = [key for key, val in affect_freqs.items() if val == max(affect_freqs.values())]

    return max_keys
