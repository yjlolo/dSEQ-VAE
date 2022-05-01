from typing import Sequence, Literal
import random
from pathlib import Path
import argparse
import subprocess

import pretty_midi
from pretty_midi import constants

from src.utils.util import ensure_dir

SAMPLE_RATE = 16000

FILE_EXT = 'mid'
SOUNDFOUNT = 'data/MuseScore_General.sf3'
INSTRUMENT_MAP = {
    'ap': 'Acoustic Grand Piano',
    'ep': 'Electric Grand Piano',
    'co': 'Church Organ',
    'guitar': 'Acoustic Guitar (nylon)',
    'ps': 'Pizzicato Strings',
    'os': 'Orchestral Harp',
    'se': 'String Ensemble 1',
    'sl': 'Lead 1 (square)'
}


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def map_inst2midi(
    target_instruments: Sequence[str],
    midi_files: Sequence[str],
    mode: Literal['full', 'part']
):
    out = {}
    if mode == 'full':
        for instrument in target_instruments:
            out[instrument] = midi_files
        
    if mode == 'part':
        n_instrument = len(target_instruments)
        n_midi = len(midi_files)
        n_midi_per_instrument = int(n_midi / n_instrument)
        midi_file_chunks = list(chunks(midi_files, n_midi_per_instrument))
        for c, m in zip(target_instruments, midi_file_chunks):
            out[c] = m
    return out


def main(args):
    filepath = args.filepath
    prefix = args.prefix
    target_instruments = args.instruments
    n_rand = args.n_rand
    mode = args.mode

    midi_filepath = list(Path(filepath).glob(f'*.{FILE_EXT}'))
    print(f"Identified {len(midi_filepath)} {FILE_EXT} files.")
    if n_rand != -1:
        random.seed(888)
        midi_filepath_sampled = random.sample(midi_filepath, k=n_rand)
    else:
        midi_filepath_sampled = midi_filepath
    print((
        f"{len(set(midi_filepath_sampled))}/{len(set(midi_filepath))} "
        f"{FILE_EXT} files will be used for wav synthesis."
    ))

    dict_inst2midi = map_inst2midi(
        target_instruments,
        midi_filepath_sampled,
        mode=mode
    )

    for t in target_instruments:
        try:
            instrument_id = INSTRUMENT_MAP[t]
        except KeyError:
            t = t.lower()
            assert t in ['vibraphone', 'trumpet']
            instrument_id = t.title()

        target_midis = dict_inst2midi[t]
        for f in target_midis:
            midi_data = pretty_midi.PrettyMIDI(str(f))
            program_id = constants.INSTRUMENT_MAP.index(instrument_id)
            for i in range(len(midi_data.instruments)):
                midi_data.instruments[i].program = program_id
                source_filename = Path(f).stem
                instrument = ''.join(instrument_id.split())
                if prefix is None:
                    prefix = ''
                target_filename = \
                    Path(f'{source_filename}-{instrument}.{FILE_EXT}')
                target_dataset = Path(f'{Path(f).parents[0]}-{instrument}')
                ensure_dir(target_dataset)
                output_path = target_dataset / target_filename
                midi_data.write(str(output_path))
                print(f"{target_filename} has been saved to {output_path}.")

                wav_output = \
                    target_dataset / Path(target_filename.stem + '.wav')
                subprocess.call([
                    'fluidsynth', '-ni', SOUNDFOUNT,
                    str(output_path),
                    '-F', wav_output,
                    '-r', str(SAMPLE_RATE)]
                )


    # for f in midi_filepath_sampled:
    #     midi_data = pretty_midi.PrettyMIDI(str(f))
    
    #     for t in target_instruments:
    #         try:
    #             instrument_id = INSTRUMENT_MAP[t]
    #         except KeyError:
    #             t = t.lower()
    #             assert t in ['vibraphone', 'trumpet']
    #             instrument_id = t.title()
    #         program_id = constants.INSTRUMENT_MAP.index(instrument_id)
    #         for i in range(len(midi_data.instruments)):
    #             midi_data.instruments[i].program = program_id
    
    #         source_filename = Path(f).stem
    #         instrument = ''.join(instrument_id.split())
    #         if prefix is None:
    #             prefix = ''
    #         target_filename = Path(f'{source_filename}-{instrument}.{FILE_EXT}')
    #         target_dataset = Path(f'{Path(f).parents[0]}-{instrument}')
    #         ensure_dir(target_dataset)
    #         output_path = target_dataset / target_filename
    #         midi_data.write(str(output_path))
    #         print(f"{target_filename} has been saved to {output_path}.")

    #         wav_output = target_dataset / Path(target_filename.stem + '.wav')
    #         subprocess.call([
    #             'fluidsynth', '-ni', SOUNDFOUNT,
    #             str(output_path),
    #             '-F', wav_output,
    #             '-r', str(SAMPLE_RATE)]
    #         )


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-f', '--filepath', type=str)
    args.add_argument('-p', '--prefix', type=str, default=None)
    args.add_argument('-i', '--instruments', nargs='+')
    args.add_argument('-n', '--n_rand', type=int, default=-1)
    args.add_argument('-m', '--mode', type=str, default='full')

    args = args.parse_args()
    main(args)
