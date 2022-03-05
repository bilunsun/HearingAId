import glob
import os

def get_latest_checkpoint(directory) -> str:
    """
    Find the most recent checkpoint from a directory and returns the checkpoint's filename

    Return None if no checkpoints found.

    Checkpoints are named:
        adjective-noun-number.ckpt
    """

    checkpoints = glob.glob(os.path.join(directory, '*.ckpt'))

    max_num = 0
    max_checkpoint = None
    for checkpoint in checkpoints:
        checkpoint_num = int(checkpoint.split('-')[-1].rstrip('.ckpt'))
        if checkpoint_num > max_num:
            max_num = checkpoint_num
            max_checkpoint = checkpoint

    return max_checkpoint


if __name__ == '__main__':
    print(get_latest_checkpoint("../checkpoints"))
