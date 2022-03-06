import time
import argparse

from rolling_window import data_thread, classify, buffer_len, dq, WINDOW_TIME_S, exit_signal


def main(args):
    data_thread.start()

    infer_times = []

    for i in range(args.num_runs):
        raw_data = list(dq)  # The 'lock' object does not seem to be necessary for reading

        if len(raw_data) < buffer_len:
            time.sleep(WINDOW_TIME_S)
            continue

        start = time.time()
        classify(raw_data)
        end = time.time()

        infer_times.append(end-start)

    avg_infer = sum(infer_times[1::]) / (len(infer_times) - 1)
    print(f'Average Inference Time: {avg_infer * 1000:.3f} ms')

    exit_signal.set()
    data_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=1000)

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        exit_signal.set()
        data_thread.join()
        print(e)
