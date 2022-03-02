# Mostly taken from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
import argparse

try:
    import librosa
except ImportError:
    librosa = False
    print(
        "Module librosa not found. Proceeding without.\n"
        + "If you are on the Jetson Nano/Raspberry Pi, ignore this warning."
    )

import matplotlib.pyplot as plt
import torch
import torchaudio
import numpy as np


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show()


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)

    if num_channels == 1:
        axes = [axes]

    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)

        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")

        if xlim:
            axes[c].set_xlim(xlim)

    figure.suptitle(title)
    plt.show()


def plot_spectrogram(spectrogram, title=None, ylabel="Frequency bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    if librosa:
        im = axs.imshow(librosa.power_to_db(spectrogram), origin="lower", aspect=aspect)
    else:
        im = axs.imshow(power_to_db(spectrogram), origin="lower", aspect=aspect)

    if xmax:
        axs.set_xlim((0, xmax))

    fig.colorbar(im, ax=axs)
    plt.show()


def plot_melspectrogram(waveform, sample_rate, n_fft=1024, hop_length=512, win_length=None, n_mels=64, norm="slaney"):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels, norm=norm
    )
    spectrogram = mel_spectrogram(waveform)  # Remove batch dimension
    if len(spectrogram.shape) == 4:
        spectrogram = spectrogram.squeeze(0)

    assert len(spectrogram.shape) == 3, "Must be 3D"

    # spectrogram = spectrogram.permute(1, 2, 0)
    spectrogram = spectrogram.mean(dim=0)
    spectrogram /= spectrogram.max()

    print("img_shape", spectrogram.shape)
    plot_spectrogram(spectrogram, title="MelSpectrogram")

    return spectrogram


def plot_x_distribution(x):
    x = torch.clip(x, min=1e-10).flatten()
    _, (ax_og, ax_log, ax_scaled) = plt.subplots(1, 3)

    ax_og.hist(x.numpy(), bins=50)
    ax_og.set_title("OG")

    x = torch.log10(x)
    ax_log.hist(x.numpy(), bins=50)
    ax_log.set_title("Log")

    x = (x - x.mean()) / x.std()
    ax_scaled.hist(x.numpy(), bins=50)
    ax_scaled.set_title("Scaled")

    plt.show()


def plot_pitch(waveform, sample_rate):
    pitch = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate)

    _, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sample_rate
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

    axis2.legend(loc=0)
    plt.show()


def plot_kaldi_pitch(waveform, sample_rate, pitch, nfcc):
    _, axis = plt.subplots(1, 1)
    axis.set_title("Kaldi Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sample_rate
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    ln1 = axis.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")
    axis.set_ylim((-1.3, 1.3))

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, nfcc.shape[1])
    ln2 = axis2.plot(time_axis, nfcc[0], linewidth=2, label="NFCC", color="blue", linestyle="--")

    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]
    axis.legend(lns, labels, loc=0)
    plt.show()


def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


def power_to_db(S, *, ref=1.0, amin=1e-10, top_db=80.0):
    S = np.asarray(S)

    magnitude = S
    ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def main(args):
    waveform, sample_rate = torchaudio.load(args.wav_path)

    print_stats(waveform, sample_rate=sample_rate)

    if args.plot_waveform:
        plot_waveform(waveform, sample_rate)

    if args.plot_pitch:
        plot_pitch(waveform, sample_rate)

    if args.plot_spectrogram:
        plot_specgram(waveform, sample_rate)

    # Update sample rate if needed
    if sample_rate != args.target_sample_rate:
        resample = torchaudio.transforms.Resample(sample_rate, args.target_sample_rate)
        waveform = resample(waveform)

    waveform = waveform[:, : args.target_sample_rate]
    mel_spectrogram = plot_melspectrogram(
        waveform,
        args.target_sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        norm=args.norm,
    )
    plot_x_distribution(mel_spectrogram)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--wav_path", type=str, default="data/UrbanSound8K/audio/fold1/7061-6-0-0.wav")

    # Choose what to plot
    parser.add_argument("--plot_waveform", action="store_true")
    parser.add_argument("--plot_pitch", action="store_true")
    parser.add_argument("--plot_spectrogram", action="store_true")

    # For MelSpectrogram
    parser.add_argument("--target_sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--win_length", type=int, default=None)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--norm", type=str, default=None)

    args = parser.parse_args()

    main(args)
