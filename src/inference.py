
import time
import torch
import torchaudio.transforms
from train import Model


TARGET_SAMPLE_RATE = 22050
N_SAMPLES = 22050
TO_MELSPECTROGRAM = torchaudio.transforms.MelSpectrogram(sample_rate=TARGET_SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)

# Raspberry Pi 4 quantized engine
import platform
if platform.system() == 'Linux' and not torch.cuda.is_available():
    torch.backends.quantized.engine = 'qnnpack'

model = Model.load_from_checkpoint("src/checkpoints/autumn-wind-80.ckpt")
model, scaler = model.model, model.scaler

# Quantization aware training
model.eval()
model_int8 = torch.quantization.convert(model)

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



t_data, sample_rate = torchaudio.load("data/UrbanSound8K/audio/fold1/7061-6-0-0.wav")
resample = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)

preprocess_times = []
infer_times = []

for i in range(100):
    start = time.time()
    x = preprocess(t_data, sample_rate, resample)
    x = scaler.transform(x)
    mid = time.time()
    model_int8(x)
    end = time.time()
    infer_times.append(end - mid)
    preprocess_times.append(mid - start)

avg_preprocess = sum(preprocess_times) / len(preprocess_times)
avg_infer = sum(infer_times) / len(infer_times)

print(f'Average Preprocessing Time: {avg_preprocess * 1000:.3f} ms')
print(f'Average Inference Time: {avg_infer * 1000:.3f} ms')
print(f'Average Total Time: {(avg_infer + avg_preprocess) * 1000:.3f} ms')
