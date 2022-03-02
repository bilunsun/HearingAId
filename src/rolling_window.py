import pyaudio
import time
import torch
import torch.nn.functional as F
from collections import deque
from threading import Event, Lock, Thread

from train import Model


CHANNELS = 1
CHUNK = 1_600
SAMPLE_RATE = 16_000
WINDOW_TIME_S = 4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model.load_from_checkpoint("checkpoints/fearless-spaceship-160.ckpt")
model = model.to(device)
model.scaler.to(device)
model = model.eval()


classes = [
    "acoustic_guitar",
    "alarm_clock",
    "bell",
    "bird",
    "brass_instrument",
    "car_alarm",
    "cat",
    "dog",
    "doorbell",
    "drum_kit",
    "explosion",
    "helicopter",
    "honking",
    "laughter",
    "plucked_string_instrument",
    "police_siren",
    "rapping",
    "reversing_beeps",
    "silence",
    "singing",
    "speech",
    "telephone_ring",
    "train_horn",
    "water",
]


def yield_data(dq: deque, lock: Lock, exit_signal: Event):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

    while not exit_signal.is_set():
        with lock:
            dq.append(stream.read(CHUNK))

    stream.stop_stream()
    stream.close()
    p.terminate()


@torch.no_grad()
def classify(x: deque):
    x = torch.frombuffer(b"".join(x), dtype=torch.int16)

    x = x.reshape(1, 1, 1, -1).float().to(device)
    x = model.scaler.transform(x)
    x = model(x)
    x = F.softmax(x, dim=1)

    max_index = torch.argmax(x, dim=1).item()
    print(classes[max_index])


# Putting as globals to kill on exit
# Gross.
lock = Lock()
buffer_len = int(SAMPLE_RATE / CHUNK * WINDOW_TIME_S)
dq = deque(maxlen=buffer_len)
exit_signal = Event()
data_thread = Thread(target=yield_data, args=(dq, lock, exit_signal,))


def main():
    data_thread.start()

    while True:
        raw_data = list(dq)  # The 'lock' object does not seem to be necessary for reading

        if len(raw_data) < buffer_len:
            time.sleep(WINDOW_TIME_S)
            continue

        classify(raw_data)


if __name__ == "__main__":
    try:
        main()
    except:
        exit_signal.set()
        data_thread.join()
