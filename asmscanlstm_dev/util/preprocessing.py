import random

import numpy as np


# aa composition: https://web.expasy.org/docs/relnotes/relstat.html
POPULATION = ["A", "Q", "L", "S", "R", "E", "K", "T", "N", "G", "M", "W", "D", "H", "F", "Y", "C", "I", "P", "V"]
WEIGHTS = [.0825, .0393, .0965, .0664, .0553, .0672, .0580, .0535, .0406, .0707, .0241, .0110, .0546, .0227, .0386, .0292, .0138, .0591, .0474, .0686]

def pad_protein_sequences(sequences: np.ndarray[np.str_], maxlen: int) -> np.ndarray[np.str_]:
    if maxlen <= 0:
        return sequences

    for i, seq in enumerate(sequences):
        k = maxlen - len(seq)
        if k > 0:
            aa_seq = "".join(aa for aa in random.choices(POPULATION, WEIGHTS, k=k))
            if random.random() < .5:
                sequences[i] = sequences[i] + aa_seq
            else:
                sequences[i] = aa_seq + sequences[i]
            
    return sequences
