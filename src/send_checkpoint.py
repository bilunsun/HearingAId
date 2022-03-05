import socket
import tqdm
import os
from lib.checkpoint_server import BUFFER_SIZE, SERVER_PORT


def send_checkpoint(checkpoint, host, port=SERVER_PORT):
    s = socket.socket()
    s.connect((host, port))

    s.send(f'{os.path.basename(checkpoint)}'.encode())
    if not os.path.isfile(checkpoint):
        raise OSError(f'File {checkpoint} not found.')

    filesize = os.path.getsize(checkpoint)
    progress = tqdm.tqdm(range(filesize), f"Transfer Progress", unit="B", unit_scale=True, unit_divisor=1024)
    with open(checkpoint, 'rb') as f:
        while bytes_read := f.read(BUFFER_SIZE):
            s.sendall(bytes_read)
            progress.update(len(bytes_read))
    print(f'Sent checkpoint {checkpoint}')
    s.close()


if __name__ == '__main__':
    send_checkpoint('./checkpoints/worthy-monkey-70.ckpt', host="localhost")
    send_checkpoint('./checkpoints/golden-armadillo-63.ckpt', host="localhost")
