import pyaudio
import sys
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

CLASSIFY_RATE = 5                   # classification rate in Hz
CLASSIFY_PERIOD = 1 / CLASSIFY_RATE
CLASSIFY_HIST_TIME = 2              # seconds of classifications to hold on to
SEND_DEBOUNCE = 20                  # only send again if 20 seconds have passed

PROBABILITY_CUTOFF = 0.4            # don't consider a classification as valid unless it's above this

PRINT_DEBUG = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model.load_from_checkpoint("model.ckpt")
model = model.to(device)
model.scaler.to(device)
model = model.eval()

print("=============================")
print("RUN NAME", model.run_name)
print("=============================")

PLATFORM = platform.system()

classes = sorted(list(model.class_names.values()))  # Should already be sorted, but just in case
assert len(classes) == model.hparams.n_classes
max_str_len = max(len(c) for c in classes)

class_to_byte = {}

for i, c in enumerate(classes):
    class_to_byte[c] = i


def yield_data(audio_data_buffer: deque, lock: Lock, exit_signal: Event):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

    while not exit_signal.is_set():
        with lock:
            audio_data_buffer.append(stream.read(CHUNK))

    stream.stop_stream()
    stream.close()
    p.terminate()

classification_to_send = ''

def send_classifications():
    global classification_to_send

    while ( 1 ):
        detection_ready.wait()

        if ( exit_signal.is_set() ):
            break
        
        class_byte = class_to_byte[classification_to_send]

        if PLATFORM == 'Linux':
            send_class(class_byte)
        else:
            # print that we did a send for Windows systems
            print(f'I2C Send -> Class: {classification_to_send}, Int: {class_byte}', flush=True)

        detection_ready.clear()


def filter_list_of_detections( detect_q ):
        MIN_DETECT_COUNT = CLASSIFY_RATE

        all_detections = list(detect_q)
        freqs = {}
        print(all_detections[-1], flush=True)

        for detection_class in all_detections:
            if detection_class in freqs:
                freqs[detection_class] += 1
            else:
                freqs[detection_class] = 1

        max_count = 0
        most_detected_class = None
        for detection_class in freqs:
            if max_count < freqs[detection_class]:
                max_count = freqs[detection_class]
                most_detected_class = detection_class

        if max_count > MIN_DETECT_COUNT and most_detected_class != "nothing": # buffer is initially filled with "nothing"s
            # send result
            return most_detected_class

        else:
            return None


prev_filtered_class = ''
prev_sent_time = 0

@torch.no_grad()
def classify(x: deque):
    global prev_filtered_class
    global classification_to_send
    global prev_sent_time

    x = torch.frombuffer(b"".join(x), dtype=torch.int16)

    x = x.reshape(1, 1, 1, -1).float().to(device)
    x /= 2**15
    x = model.scaler.transform(x)
    x = model(x).squeeze(0)
    x = F.softmax(x, dim=0)

    max_prob, max_index = torch.max(x, dim=0)

    if max_prob > PROBABILITY_CUTOFF:
        detect_q.append(classes[max_index])
    else:
        detect_q.append("nothing")

    filtered_class = filter_list_of_detections( detect_q )
    if ( filtered_class != prev_filtered_class or \
    time.time() - prev_sent_time > SEND_DEBOUNCE ) and \
    filtered_class != None:
        prev_filtered_class = filtered_class
        prev_sent_time = time.time()

        classification_to_send = filtered_class
        detection_ready.set()

    if PRINT_DEBUG:
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
audio_data_buffer = deque(maxlen=buffer_len)
exit_signal = Event()
data_thread = Thread(target=yield_data, args=(audio_data_buffer, lock, exit_signal,))

# Queue to store detection results
detect_q_len = CLASSIFY_RATE * CLASSIFY_HIST_TIME
detect_q = deque(["nothing" for x in range(detect_q_len)], maxlen=detect_q_len)  # stuff full of empty string
detection_ready = Event() # global event for signalling send_thread
send_thread = Thread(target=send_classifications)


def main():

    detection_ready.clear()

    data_thread.start()
    send_thread.start()

    last_classification_time = time.time()
    while True:

        if time.time() - last_classification_time >= CLASSIFY_PERIOD:
            raw_data = list(audio_data_buffer)

            if len(raw_data) < buffer_len:
                time.sleep(WINDOW_TIME_S)
                continue

            classify(raw_data)
            last_classification_time = time.time()
        else:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit_signal.set()
        detection_ready.set()

        send_thread.join()
        data_thread.join()
    except Exception as e:
        # other error
        exit_signal.set()
        detection_ready.set()

        send_thread.join()
        data_thread.join()
        print(e)
