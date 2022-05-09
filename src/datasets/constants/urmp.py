SR = 16000 
NFFT = 2048
HOP = 256
NMEL = 80
SEQ_LEN = int((4 * SR) / HOP)

INSTRUMENTS = (
    "vn",
    "fl",
    "tpt",
    "cl",
    "sax"
)

DICT_INST_TO_IDX = {
    instrument: n for n, instrument in enumerate(INSTRUMENTS)
}

DICT_IDX_TO_INST = {v: k for k, v in DICT_INST_TO_IDX.items()}
