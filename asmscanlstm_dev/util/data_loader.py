import numpy as np
from Bio import SeqIO


def load_data(neg_fasta_filepath: str, pos_fasta_filepath: str) -> tuple[np.ndarray[np.str_], np.ndarray[np.int32]]:
    _, neg = read_fasta(neg_fasta_filepath)
    _, pos = read_fasta(pos_fasta_filepath)

    x = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])

    index = shuffle_index(len(x))
    return x[index], y[index]

def shuffle_index(indexes_number: int) -> np.ndarray[np.int32]:
    index = np.arange(indexes_number)
    np.random.shuffle(index)
    return index

def read_fasta(fasta_filepath: str) -> tuple[np.ndarray[np.str_], np.ndarray[np.str_]]:
    ids = []
    seqs = []

    for record in SeqIO.parse(fasta_filepath, "fasta"):
        ids.append(record.id)
        seqs.append(str(record.seq))

    return np.asarray(ids), np.asarray(seqs)
