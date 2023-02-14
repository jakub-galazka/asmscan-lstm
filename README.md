# ASMscan-LSTM

![GitHub License](https://img.shields.io/github/license/jakub-galazka/asmscan-lstm)

Bidirectional LSTM model for detection of amyloid signaling motifs (ASMs).

## Installation

```bash
pip install asmscan-lstm
```

## Usage

```python
from asmscanlstm import ASMscanLSTM

aa_seqs = [
    "MEGRASGSARIYQAGGDQYIEE",
    "VSLRAGAHDGGRIYQAVGDQYIYE",
    "HASGHGRVFQSAGDQHITEH"
]

model = ASMscanLSTM()
pred, frags = model.predict(aa_seqs)
```

## References

ASMscan-LSTM model is part of the [ASMscan](https://github.com/wdyrka-pwr/ASMscan) project:

* Not yet published.

ASMscan project is an extension of the ASMs analysis conducted with the [PCFG-CM](https://git.e-science.pl/wdyrka/pcfg-cm) model:

* W. Dyrka, M. Gąsior-Głogowska, M. Szefczyk, N. Szulc, "Searching for universal model of amyloid signaling motifs using probabilistic context-free grammars", *BMC Bioinformatics*, 22, 222, 2021.

* W. Dyrka, M. Pyzik, F. Coste, H. Talibart, "Estimating probabilistic context-free grammars for proteins using contact map constraints", *PeerJ*, 7, e6559, 2019.
