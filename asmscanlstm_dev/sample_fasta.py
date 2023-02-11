import sys

import numpy as np
from Bio import SeqIO


def sample_fasta(fasta_filepath: str, target_size: int) -> None:
    records = list(SeqIO.parse(fasta_filepath, "fasta"))

    if target_size >= len(records):
        raise OverflowError("The number of samples cannot be greater than or equal to the size of the sampled collection!")

    index = np.random.choice(len(records), target_size, False)
    
    sample = []
    for i in index:
        sample.append(records[i])

    with open(fasta_filepath.split(".")[0] + f"_sampled{target_size}.fa", "w") as f:
        SeqIO.write(sample, f, "fasta")

if __name__ == "__main__":
    sample_fasta(sys.argv[1], int(sys.argv[2]))
