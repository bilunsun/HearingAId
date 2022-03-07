
import argparse
import os
import pyaudio
import time
import uuid
import wave


CHANNELS = 1
FORMAT = pyaudio.paInt16


def main(args):
    # Calculate the necessary buffer length
    buffer_len = int(args.sample_rate / args.chunk * args.window_time_s)

    # Create output dir if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create audio object
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=args.sample_rate, input=True, frames_per_buffer=args.chunk)

    input("Start")

    # Collect data
    for _ in range(args.n_files):
        if args.manual:
            input(f"Press enter to start recording {args.window_time_s}s of audio for class '{args.class_label}'")
            time.sleep(0.1)  # Slight delay to avoid recording the initial key press

        frames = [stream.read(args.chunk) for _ in range(buffer_len)]

        # Save to file
        unique_id = str(uuid.uuid4()).split("-")[0]
        output_filename = os.path.join(args.output_dir, f"{args.class_label}_{unique_id}.wav")
        wf = wave.open(output_filename, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(args.sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close()

        print(f"Saved to {output_filename}\n")

    # Teardown
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("class_label", type=str)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--n_files", type=int, default=5)
    parser.add_argument("--chunk", type=int, default=1_600)
    parser.add_argument("--sample_rate", type=int, default=16_000)
    parser.add_argument("--window_time_s", type=float, default=2)
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "collected_data"))

    args = parser.parse_args()

    main(args)
