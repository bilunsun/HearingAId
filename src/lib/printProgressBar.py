import shutil


def printProgressBar(iteration, total, prefix='', suffix='', fill='█',
                     barfill=' ', printEnd="\r"):
    w, h = shutil.get_terminal_size()
    length = w - 9 - len(str(total)) * 2

    percent = f'{int(100 * (iteration / total)) : >3}'
    filledLength = int(length * iteration // total)

    bar = fill * filledLength + barfill * (length - filledLength)

    print(f'\r{percent}%|{bar}| {iteration}/{total}', end=printEnd, flush=True)

    if iteration == total:
        print()
