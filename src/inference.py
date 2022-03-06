import time
import torch
import torchaudio.transforms
from train import Model
import platform
import argparse

TARGET_SAMPLE_RATE = 22050
N_SAMPLES = 22050
TO_MELSPECTROGRAM = torchaudio.transforms.MelSpectrogram(sample_rate=TARGET_SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)


# Raspberry Pi 4/Jetson Nano quantized engine
if platform.system() == 'Linux':
    torch.backends.quantized.engine = 'qnnpack'


def preprocess(t_data, sample_rate, resample):

    # Reshape
    if len(t_data.shape) == 1:
        t_data = t_data.unsqueeze(1)
    elif t_data.size(1) == 1 or t_data.size(1) == 2:
        t_data = t_data.T

    # Resample
    t_data = resample(t_data)

    # Mixdown
    if t_data.shape[0] > 1:
        t_data = torch.mean(t_data, dim=0, keepdim=True)

    t_data = t_data[:, :N_SAMPLES]

    # Melspectrogram
    t_data = TO_MELSPECTROGRAM(t_data)

    # Create batch dimension
    t_data = t_data.unsqueeze(0)

    return t_data


def main(args):
    global TO_MELSPECTROGRAM

    model = Model.load_from_checkpoint(args.checkpoint)
    model, scaler = model.model, model.scaler

    # Quantization aware training
    if args.use_quant:
        model.eval()
        model = torch.quantization.convert(model)

    # Move unquantized model to CUDA if available
    if args.use_cuda:
        if torch.cuda.is_available():
            model.to('cuda')
        else:
            raise RuntimeError('CUDA is not available on this device')

    model_type = 'Quantized' if args.use_quant else 'Unquantized'
    model_loc = 'CUDA' if args.use_cuda else 'CPU'
    print(f'\nBenchmark is running {model_type} on {model_loc}\n')

    t_data, sample_rate = torchaudio.load("data/UrbanSound8K/audio/fold1/7061-6-0-0.wav")
    resample = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)

    if args.use_cuda:
        t_data = t_data.to('cuda')
        resample = resample.to('cuda')
        TO_MELSPECTROGRAM = TO_MELSPECTROGRAM.to('cuda')

    preprocess_times = []
    infer_times = []

    for i in range(args.n_runs):
        start = time.time()
        x = preprocess(t_data, sample_rate, resample)
        x = scaler.transform(x)
        mid = time.time()
        model(x)
        end = time.time()
        infer_times.append(end - mid)
        preprocess_times.append(mid - start)

    if args.use_cuda and args.n_runs != 1:
        avg_preprocess = sum(preprocess_times[1::]) / (len(preprocess_times) - 1)
        avg_infer = sum(infer_times[1::]) / (len(infer_times) - 1)
        print('Note: due to initialization of CUDA, first run is dropped from averages')
    else:
        avg_preprocess = sum(preprocess_times) / len(preprocess_times)
        avg_infer = sum(infer_times) / len(infer_times)

    print(f'Average Preprocessing Time: {avg_preprocess * 1000:.3f} ms')
    print(f'Average Inference Time: {avg_infer * 1000:.3f} ms')
    print(f'Average Total Time: {(avg_infer + avg_preprocess) * 1000:.3f} ms')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--checkpoint', default="src/checkpoints/autumn-wind-80.ckpt")
    argparser.add_argument('--use_cuda', action='store_true')
    argparser.add_argument('--n_runs', default=100, type=int)
    argparser.add_argument('--use_quant', action='store_true')

    args = argparser.parse_args()

    main(args)
