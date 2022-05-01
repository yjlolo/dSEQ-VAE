SR = 16000 
NFFT = 2048
HOP = 256
NMEL = 80
SEQ_LEN = int((4 * SR) / HOP)

DICT_INST_TO_IDX = {
    'AcousticGrandPiano': 0,
    'ChurchOrgan': 1,
    'AcousticGuitar(nylon)': 2,
    'Violin': 3,
    'Trumpet': 4,
}

DICT_IDX_TO_INST = {v: k for k, v in DICT_INST_TO_IDX.items()}
