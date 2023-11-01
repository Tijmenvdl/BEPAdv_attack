'''
Module that contains all functions relevant for analysing data w.r.t. the NRC Emotions quantifier
'''

from nrclex import NRCLex

nrc = NRCLex("Do you wanna know how I got these scars? My father was a drinker...and a fiend. And one night, he goes off crazier than usual. Mommy gets the kitchen knife to defend herself. He doesn't like that. Not...one...bit. So, me watching, he takes the knife to her, laughing while he does it. He turns to me, and he says, Why so serious? He comes at me with the knife - Why so serious? He sticks the blade in my mouth - Let's put a smile on that face! And... why so serious?")

print(nrc.raw_emotion_scores)
print(nrc.top_emotions)
print(nrc.affect_frequencies)

import pandas as pd

word_embeddings = pd.read_csv("word_embeddings/glove.6B.300d.txt", header=None, sep='\t')

