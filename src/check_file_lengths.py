import argparse
import os

import seaborn as sns
import matplotlib.pyplot as plt
import torchaudio

import pandas as pd

from pathlib import Path
from typing import Union


CLASSID_TO_NAME = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music'
}


def get_wav_length_seconds(fname: Union[str, Path]) -> float:
    """
    Return length of a file name in seconds. Returns -1 if file cannot be opened or found.

    Raises TypeError if fname does not have a .wav extension

    :param fname: Path to .wav file
    :return: length of .wav file in seconds (float)
    """

    _, ext = os.path.splitext(fname)

    if ext.lower() != '.wav':
        raise TypeError(f'File {fname} is of invalid type. Expected: .wav, received: {ext}')

    metadata = torchaudio.info(str(fname))
    return metadata.num_frames / metadata.sample_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recursively check audio file lengths in a directory.')
    parser.add_argument('dir', type=str, help='Directory to check (recursive)')
    parser.add_argument('--min', type=float, help='Minimum file length allowed', default=0.5)
    parser.add_argument('--max', type=float, help='Maximum file length allowed', default=5)
    parser.add_argument('--plot', '-p', action='store_true', help='Show histogram of lengths')
    parser.add_argument('--verbose', '-v', action='store_true', help='Output more detailed info')
    parser.add_argument('--format', '-f', choices=[None, 'UrbanSound8K'], default='UrbanSound8K')

    args = parser.parse_args()

    wav_files = list(Path(args.dir).rglob("*.[wW][aA][vV]"))
    data = []

    for wav in wav_files:
        wav_len = get_wav_length_seconds(wav)
        if (wav_len <= args.min or wav_len >= args.max) and args.verbose:
            print(f'File {wav} violates duration criteria. Length: {wav_len}')

        if args.format == 'UrbanSound8K':
            # File name format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav
            wav_classId = int(str(wav).split('-')[1])
            classname = CLASSID_TO_NAME[wav_classId]
        else:
            classname = None

        data.append((wav, wav_len, classname))

    df = pd.DataFrame(data=data, columns=['File', 'Length', 'Class'])

    if args.plot:
        sns.set_theme(style='ticks')

        f, [ax0, ax1] = plt.subplots(2, figsize=(15, 15))
        sns.despine(f)

        sns.histplot(
            df,
            x='Length', hue='Class',
            multiple='stack',
            ax=ax0
        )

        sns.countplot(
            data=df,
            x='Class',
            ax=ax1
        )

        plt.show()

    else:
        print('To show a histogram of lengths, use --plot option.')

    print('Done!')
