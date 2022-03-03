import pyaudio
import time
import torch
import torch.nn.functional as F
from collections import deque
from threading import Event, Lock, Thread
import platform

from train import Model
from user_output import send_class


CHANNELS = 1
CHUNK = 1_600
SAMPLE_RATE = 16_000
WINDOW_TIME_S = 4

CLASSIFY_RATE = 5  # classification rate in Hz
CLASSIFY_PERIOD = 1 / CLASSIFY_RATE
CLASSIFY_HIST_TIME = 2  # seconds of classifications to hold on to
SEND_DEBOUNCE = 60  # only send again if 1 minute has passed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model.load_from_checkpoint("checkpoints/fearless-spaceship-160.ckpt")
model = model.to(device)
model.scaler.to(device)
model = model.eval()

PLATFORM = platform.system()


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

class_to_byte = {}

for i, c in enumerate(classes):
    class_to_byte[c] = i


def yield_data(dq: deque, lock: Lock, exit_signal: Event):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

    while not exit_signal.is_set():
        with lock:
            dq.append(stream.read(CHUNK))

    stream.stop_stream()
    stream.close()
    p.terminate()


def send_classifications():
    """
    Applies a low-pass filter to the classes detected, and sends the classification
    """
    prev_sent = 'silence'
    prev_sent_time = time.time()

    MIN_DETECT_COUNT = CLASSIFY_RATE

    while not exit_signal.is_set():
        # dumb low pass filter
        # look at the past history. If class persists for more than 1 second and is the most frequent
        detected = list(detect_q)
        freqs = {}

        for c in detected:
            if c in freqs:
                freqs[c] += 1
            else:
                freqs[c] = 1

        max_count = 0
        max_class = None
        for c in freqs:
            if max_count < freqs[c]:
                max_count = freqs[c]
                max_class = c

        if max_count > MIN_DETECT_COUNT and \
           max_class != 'silence' and \
           max_class != prev_sent and \
           time.time() - prev_sent_time > SEND_DEBOUNCE:  # noqa: E125
            # send result
            if PLATFORM == 'Linux':
                send_class(class_to_byte[max_class])
            else:
                # print that we did a send for Windows systems
                print(f'I2C Send -> Class: {max_class}, Int: {class_to_byte[max_class]}')

            # send class once
            prev_sent = max_class

        time.sleep(1)


@torch.no_grad()
def classify(x: deque):
    x = torch.frombuffer(b"".join(x), dtype=torch.int16)

    x = x.reshape(1, 1, 1, -1).float().to(device)
    x = model.scaler.transform(x)
    x = model(x)
    x = F.softmax(x, dim=1)

    max_index = torch.argmax(x, dim=1).item()
    print(classes[max_index], x)
    with detect_q_lock:
        detect_q.append(classes[max_index])


# Putting as globals to kill on exit
# Gross.
lock = Lock()
buffer_len = int(SAMPLE_RATE / CHUNK * WINDOW_TIME_S)
dq = deque(maxlen=buffer_len)
exit_signal = Event()
data_thread = Thread(target=yield_data, args=(dq, lock, exit_signal,))

# Queue to store detection results
detect_q_lock = Lock()
detect_q_len = CLASSIFY_RATE * CLASSIFY_HIST_TIME
detect_q = deque(["silence" for x in range(detect_q_len)], maxlen=detect_q_len)  # stuff full of silence
send_thread = Thread(target=send_classifications)


def main():
    data_thread.start()
    send_thread.start()

    last_classification = time.time()
    while True:
        raw_data = list(dq)  # The 'lock' object does not seem to be necessary for reading

        if len(raw_data) < buffer_len:
            time.sleep(WINDOW_TIME_S)
            continue

        if time.time() - last_classification >= CLASSIFY_PERIOD:
            classify(raw_data)
            last_classification = time.time()
        else:
            pass


if __name__ == "__main__":
    try:
        main()
    except:
        exit_signal.set()
        data_thread.join()
