import gradio as gr
import torch
import torchaudio.transforms
import torch.nn.functional as F
import tempfile
from train import Model


TARGET_SAMPLE_RATE = 22050
N_SAMPLES = 22050
TO_MELSPECTROGRAM = torchaudio.transforms.MelSpectrogram(sample_rate=TARGET_SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=128)
CLASS_IDS = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]

model = Model.load_from_checkpoint("checkpoints/youthful-lake-45.ckpt")


def preprocess(audio):
    if isinstance(audio, tempfile._TemporaryFileWrapper):
        audio = torch.load(audio.name)
        return audio.unsqueeze(0)

    sample_rate, data = audio
    t_data = torch.from_numpy(data)

    # Reshape
    if len(t_data.shape) == 1:
        t_data = t_data.unsqueeze(1)
    t_data = t_data.transpose(0, 1).float()

    # Normalize
    t_data = t_data / t_data.max()

    # Resample
    if sample_rate != TARGET_SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
        t_data = resample(t_data)

    # Mixdown
    if t_data.shape[0] > 1:
        t_data = torch.mean(t_data, dim=0, keepdim=True)

    # Fix length
    len_data = t_data.shape[1]
    if len_data > N_SAMPLES:
        t_data = t_data[:, :N_SAMPLES]
    else:
        len_missing = N_SAMPLES - len_data
        t_data = F.pad(t_data, (0, len_missing))

    # Melspectrogram
    t_data = TO_MELSPECTROGRAM(t_data)

    # Create batch dimension
    t_data = t_data.unsqueeze(0)

    return t_data


@torch.no_grad()
def audio_classification(audio1, audio2, audio3):
    audio = audio1 or audio2 or audio3
    t_data = preprocess(audio)

    x = model.scaler.transform(t_data)
    logits = model(x).flatten()
    preds = F.softmax(logits, dim=0)

    return dict(zip(CLASS_IDS, preds.tolist()))


inputs = [
    gr.inputs.Audio(source="upload", type="numpy", label="WAV", optional=True),
    gr.inputs.Audio(source="microphone", type="numpy", label="Record", optional=True),
    gr.inputs.File(file_count="single", type="file", label="tensor", optional=True)
]
outputs = gr.outputs.Label(num_top_classes=3)
interface = gr.Interface(fn=audio_classification, inputs=inputs, outputs=outputs).launch()
