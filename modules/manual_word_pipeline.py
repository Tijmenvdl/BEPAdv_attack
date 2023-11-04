'''
Module contains functions and class ManualAttack for from-scratch approach on string-level (not df-level).
'''

#Imports
import language_tool_python

from modules.nrc_tool import nrc_affect_dict, nrc_affect_freqs, nrc_top_emotions

#Functions and class
def lang_check(str1_, str2_, lang_tool_):
    '''
    Functions performs language/grammar check on two strings.
    Will fix in second string only the fixes that are found on top of the ones found in first string.
    Parameters:
        -str1_: First string
        -str2_: Second string with one word-level substitution compared to first
        -lang_tool_: Loaded language check tool
    Returns:
        -str2_fixed: Grammar-fixed second string, using only the fixes found on top of the first string's ones.
    '''

    # Find language errors in both string
    matches_1, matches_2 = lang_tool_.check(str1_), lang_tool_.check(str2_)

    # Look for that one fix
    found_fixes = []
    while not found_fixes:
        try:
            # Find the inserted error
            # Only if there is a higher amount of errors in the new string, a correction must take place.
            if len(matches_1) < len(matches_2):
                for match_2 in enumerate(matches_2):
                    if match_2[1].message != matches_1[match_2[0]].message:
                        found_fixes.append(match_2[1])
            # In any other case or if found_fixes stays empty, the while-loop is aborted
            break
        # In case the previous loop leads to an indexerror, we have arrived at the final message
        except IndexError:
            found_fixes.append(matches_2[-1])

    str2_fixed = language_tool_python.utils.correct(str2_, found_fixes)
    return str2_fixed

class ManualAttack:
    '''
    Class ManualAttack containing all functions needed for manual word-level substitution strategies.
    '''

    def __init__(self, text_, embeddings_, ):
        self.text = text_
        self.embeddings = embeddings_
