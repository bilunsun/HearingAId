import numpy as np
import pyaudio
import time
import torch
import torchaudio
from collections import deque
from threading import Event, Lock, Thread

from lib.audio_utils import plot_melspectrogram


CHANNELS = 1
CHUNK = 1024
RATE = 44100
TARGET_SAMPLE_RATE = 22050
WINDOW_TIME_S = 1


def yield_data(dq: deque, lock: Lock, exit_signal: Event):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    for _ in range(1000000):  # Test without infinite loop for now
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


# Putting as globals to kill on exit
# Gross.
lock = Lock()
buffer_len = int(RATE / CHUNK * WINDOW_TIME_S)
dq = deque(maxlen=buffer_len)
exit_signal = Event()
data_thread = Thread(target=yield_data, args=(dq, lock, exit_signal,))


def main():
    data_thread.start()

    # Countdown
    print("Starting in:")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("Recording...")

    # Recording
    all_data = []
    for _ in range(10):
        raw_data = list(dq)  # The 'lock' object does not seem to be necessary for reading

        if len(raw_data) < buffer_len:
            time.sleep(2)
            continue

        # visualize(raw_data)
        # curr_time = time.time()
        # print(curr_time % 1, "OK")
        # time.sleep(0.1)
        all_data.append(raw_data)
        time.sleep(0.1)
    print("DONE.")

    # Visualizing
    for data in all_data:
        visualize(data)

    exit_signal.set()
    data_thread.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit_signal.set()
        data_thread.join()
