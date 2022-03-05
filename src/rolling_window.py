import pyaudio
import sys
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
model = Model.load_from_checkpoint("checkpoints/super-dream-174.ckpt")
model = model.to(device)
model.scaler.to(device)
model = model.eval()


classes = [
    "doorbell",
    "honking",
    "knocking",
    # "silence",
    "siren",
    # "talking",
]
# assert len(classes) == model.hparams.n_classes
max_str_len = max(len(c) for c in classes)


def yield_data(dq: deque, lock: Lock, exit_signal: Event):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

    while not exit_signal.is_set():
        with lock:
            dq.append(stream.read(CHUNK))

    stream.stop_stream()
    stream.close()
    p.terminate()


past_classifications = deque(maxlen=100)
previous_classification = None
@torch.no_grad()
def classify(x: deque):
    global past_classifications
    global previous_classification
    global console

    x = torch.frombuffer(b"".join(x), dtype=torch.int16)

    x = x.reshape(1, 1, 1, -1).float().to(device)
    x /= 2**15
    x = model.scaler.transform(x)
    x = model(x)
    x = F.softmax(x, dim=1)

    max_index = torch.argmax(x, dim=1).item()

    # Simple filtering
    past_classifications.append(max_index)
    filtered_max = max(set(past_classifications), key=past_classifications.count)
    if filtered_max != previous_classification:
        previous_classification = filtered_max

    repr_str = ""
    for c, prob in zip(classes, x.flatten()):
        prob_len = int(prob.item() * 100)
        remaining_len = 100 - prob_len
        prob_bar = "#" * prob_len + "-" * remaining_len
        repr_str += f"{c: <{max_str_len}}: [{prob_bar}]\n"
    repr_str += "\n\n"
    print(repr_str, end="\r")
    sys.stdout.flush()
    time.sleep(0.1)


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
    except Exception as e:
        exit_signal.set()
        data_thread.join()
        raise e
    except KeyboardInterrupt:
        exit_signal.set()
        data_thread.join()
