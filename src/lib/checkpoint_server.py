import socket
import os
from threading import Event

SERVER_ADDRESS = "0.0.0.0"
SERVER_PORT = 5001
BUFFER_SIZE = 4096
CHECKPOINT_DIR = '../downloaded_checkpoints'


def checkpoint_server(exit_signal: Event, new_checkpoint_signal: Event):
    s = socket.socket()
    s.settimeout(2)
    s.bind((SERVER_ADDRESS, SERVER_PORT))

    while not exit_signal.is_set():
        s.listen()
        try:
            client_socket, address = s.accept()  # this call is blocking, will stop thread from being join()-ed
        except socket.timeout:
            pass
        else:
            received = client_socket.recv(BUFFER_SIZE).decode()
            filename = received

            filename = os.path.basename(filename)

            with open(os.path.join(CHECKPOINT_DIR, filename), 'wb') as f:
                while bytes_read := client_socket.recv(BUFFER_SIZE):
                    f.write(bytes_read)

            client_socket.close()
            new_checkpoint_signal.set()

    s.close()


if __name__ == '__main__':
    exit_signal = Event()
    new_checkpoint_signal = Event()
    checkpoint_server(exit_signal, new_checkpoint_signal)
