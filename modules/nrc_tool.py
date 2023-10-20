'''
Module that contains all functions relevant for analysing data w.r.t. the NRC Emotions quantifier
'''

from nrclex import NRCLex

nrc = NRCLex("I like relaxing by the sea.")

print(nrc.affect_list)
print(nrc.affect_dict)
print(nrc.raw_emotion_scores)
