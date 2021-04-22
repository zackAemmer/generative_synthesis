import numpy as np
import pandas as pd

# Function to turn logits returned by the model back to the OHE variables
def logit_to_ohe(logit_array):
    # Find maximum in the array and make it one, all others zero
    max_idx = np.argmax(logit_array)
    final = np.zeros(len(logit_array))
    final[max_idx] = 1
    return final

def logit_to_val(logit_array, encodings_array):
    # Find maximum logit index, return the corresponding PUMS encoding
    max_idx = np.argmax(logit_array)
    return encodings_array[max_idx]