import numpy as np
import pyaudio
import time
import torch
import torch.nn.functional as F
import torchaudio
from collections import deque
from threading import Event, Lock, Thread
import argparse

from lib.audio_utils import plot_melspectrogram
from train import Model


CHANNELS = 1
CHUNK = 1024
RATE = 44100
TARGET_SAMPLE_RATE = 16_000
WINDOW_TIME_S = 10


model = Model.load_from_checkpoint("checkpoints/fearless-spaceship-160.ckpt")
model = model.cuda()
model = model.eval()

classes = [
    "alarm_clock",
    "car_alarm",
    "doorbell",
    "honking",
    "police_siren",
    "reversing_beeps",
    "telephone_ring",
    "train_horn",
]


def yield_data(dq: deque, lock: Lock, exit_signal: Event):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    while True:  # Test without infinite loop for now
        if exit_signal.is_set():
            break

        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

        with lock:
            dq.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()


def visualize(raw_data: deque, target_sample_rate: int = TARGET_SAMPLE_RATE):
    window = np.concatenate(raw_data)
    waveform = torch.from_numpy(window).unsqueeze(0).float()

    resample = torchaudio.transforms.Resample(RATE, target_sample_rate)
    resampled_waveform = resample(waveform)
    plot_melspectrogram(resampled_waveform, target_sample_rate)


def classify(raw_data: deque, target_sample_rate: int = TARGET_SAMPLE_RATE):
    window = np.concatenate(raw_data)
    waveform = torch.from_numpy(window).reshape(1, 1, 1, -1).float()

    resample = torchaudio.transforms.Resample(RATE, target_sample_rate)
    x = resample(waveform)
    x = model.scaler.transform(x).cuda()

    with torch.no_grad():
        pred = F.softmax(model(x), dim=1)

    max_index = torch.argmax(pred, dim=1)
    print(classes[max_index], pred)


# Putting as globals to kill on exit
# Gross.
lock = Lock()
buffer_len = int(RATE / CHUNK * WINDOW_TIME_S)
dq = deque(maxlen=buffer_len)
exit_signal = Event()
data_thread = Thread(target=yield_data, args=(dq, lock, exit_signal,))


def main(args):
    data_thread.start()

    infer_times = []

    for i in range(args.num_runs):
        raw_data = list(dq)  # The 'lock' object does not seem to be necessary for reading

        if len(raw_data) < buffer_len:
            time.sleep(2)
            continue

        start = time.time()
        classify(raw_data)
        end = time.time()

        infer_times.append(end-start)

    avg_infer = sum(infer_times[1::]) / (len(infer_times) - 1)
    print(f'Average Inference Time: {avg_infer * 1000:.3f} ms')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=100)

    args = parser.parse_args()

    try:
        main(args)
    except:
        exit_signal.set()
        data_thread.join()
