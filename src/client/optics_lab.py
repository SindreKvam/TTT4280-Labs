"""This module contains methods for the optics lab."""

import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams.update(
    {
        "font.size": 12,
    }
)

logger = logging.getLogger(__name__)


def check_saturated(data: np.ndarray, threshold: int = 255) -> bool:
    """Check if the data is saturated"""

    return np.any(data >= threshold)


def check_underexposed(data: np.ndarray, threshold: int = 10) -> bool:
    """Check if the data is underexposed"""

    return np.any(data <= threshold)


def dc_block(data: np.ndarray, *, axis: int) -> np.ndarray:
    """Remove the DC component from the data"""

    return data - data.mean(axis=axis)


def calculate_fft(
    data: np.ndarray, *, fs: float, axis: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the FFT of the data"""

    _fft_data = np.fft.fftshift(np.fft.fft(data, axis=axis), axes=axis)
    _freq = np.fft.fftshift(np.fft.fftfreq(_fft_data.shape[axis], d=1 / fs), axes=axis)

    return _freq, _fft_data


def calculate_psd_from_fft(data: np.ndarray, *, fs: float, axis: int = 1) -> np.ndarray:
    """Calculate the power spectral density from the FFT

    The power spectral density is the square of the absolute value of the FFT
    divided by the number of samples and the sampling frequency.
    Giving the power of the signal in each frequency bin.
    """

    return np.abs(data) ** 2 / (data.shape[axis] * fs)


def calculate_snr_from_psd(
    frequencies: np.ndarray, data: np.ndarray, *, axis: int = 1, freq_margin: int = 0.1
) -> float:
    """Calculate the signal to noise ratio of the power spectral density per herz"""

    _peak_frequencies = frequencies[np.argmax(data, axis=axis)]
    lower_bounds = _peak_frequencies - freq_margin
    upper_bounds = _peak_frequencies + freq_margin
    _signal_mask = (frequencies[:, None] >= lower_bounds[None, :]) & (
        frequencies[:, None] <= upper_bounds[None, :]
    )

    # P_signal = sum(data)_signal_band * df
    _signal = np.sum(data * _signal_mask, axis=axis) * (frequencies[1] - frequencies[0])
    # P_noise = sum(data)_noise_band * df
    _noise = np.sum(data * ~_signal_mask, axis=axis) * (frequencies[1] - frequencies[0])

    _, ax = plt.subplots(tight_layout=True)

    masked_data = data * _signal_mask
    mask_R = masked_data[:, 0] > 0
    mask_G = masked_data[:, 1] > 0
    mask_B = masked_data[:, 2] > 0

    ax.plot(
        frequencies,
        (data)[:, 0],
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Støy R",
    )
    ax.plot(
        frequencies,
        (data)[:, 1],
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Støy G",
    )
    ax.plot(
        frequencies,
        (data)[:, 2],
        color="blue",
        linestyle="--",
        alpha=0.5,
        label="Støy B",
    )

    ax.plot(
        frequencies[mask_R],
        (data * _signal_mask)[mask_R, 0],
        linewidth=1.5,
        color="red",
        label="Signal R",
    )
    ax.plot(
        frequencies[mask_G],
        (data * _signal_mask)[mask_G, 1],
        linewidth=2,
        color="green",
        label="Signal G",
    )
    ax.plot(
        frequencies[mask_B],
        (data * _signal_mask)[mask_B, 2],
        linewidth=2,
        color="blue",
        label="Signal B",
    )

    plt.axvspan(
        lower_bounds[0],
        upper_bounds[0],
        color="red",
        alpha=0.2,
        label="Signalområde rød",
    )
    plt.axvspan(
        lower_bounds[1],
        upper_bounds[1],
        color="green",
        alpha=0.2,
        label="Signalområde grønn",
    )
    plt.axvspan(
        lower_bounds[2],
        upper_bounds[2],
        color="blue",
        alpha=0.2,
        label="Signalområde blå",
    )

    ax.set_title("Signal til støyforhold")
    ax.set_xlabel("Frekvens (Hz)")
    ax.set_ylabel("Effekt ($V^2/$Hz)")
    ax.grid()
    ax.legend()

    plt.savefig("snr.png", dpi=300)

    return 10 * np.log10(_signal / _noise)


def butter_filter(
    data: np.ndarray,
    *,
    cutoff: float,
    fs: float,
    btype: str = "low",
    axis: int = 1,
    order: int = 5,
) -> np.ndarray:
    """Apply a butterworth lowpass filter to the data"""

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = signal.butter(N=order, Wn=normal_cutoff, btype=btype, output="sos")

    return signal.sosfiltfilt(sos, data, axis=axis)


def plot_rgb_channels(
    data: np.ndarray,
    x: np.ndarray | None = None,
    title: str = "",
    xlabel: str = "Bilde",
    ylabel: str = "Intensitet",
    xscale: str = "linear",
    yscale: str = "linear",
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    line: str = "-",
) -> None:
    """Plot rgb channels"""

    if x is None:
        x = np.arange(data.shape[0])

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, data[:, 0], line, label="R", color="red")
    ax.plot(x, data[:, 1], line, label="G", color="green")
    ax.plot(x, data[:, 2], line, label="B", color="blue")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()
    ax.grid()

    return fig, ax


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data processing of optics measurements")
    parser.add_argument("filename", help="Path to filename that should be analysed")
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second of the video"
    )
    parser.add_argument(
        "--reference-light",
        help="File containing reference light, if you also want transmittance measurements.",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    raw_data = np.loadtxt(args.filename)
    # Raw data contains R,G,B values of measured video

    plot_rgb_channels(raw_data, title="Rådata", ylabel=r"Intensitet $\phi(z)$")
    plt.tight_layout()
    plt.savefig("raw_data.png", dpi=300)

    if args.reference_light:
        raw_data_light = np.loadtxt(args.reference_light)
        plot_rgb_channels(
            raw_data_light, title="Rådata lyskilde", ylabel=r"Intensitet $\phi(0)$"
        )
        plt.tight_layout()
        plt.savefig("raw_data_light_source.png", dpi=300)

        transmittance = raw_data / raw_data_light
        plot_rgb_channels(
            transmittance * 100,
            title="Transmittans",
            ylabel=r"Relativ intensitet $\frac{\phi(z)}{\phi(0)}$ [%]",
        )
        plt.tight_layout()
        plt.savefig("transmittance.png", dpi=300)

        transmittance_value = np.mean(transmittance, axis=0)
        print(f"Transmittance value: {transmittance_value}")

    # Check if any of the channels are saturated or underexposed
    usable_data = np.zeros(raw_data.shape)
    for i, channel in enumerate(["R", "G", "B"]):

        if check_saturated(raw_data[:, i], threshold=250):
            print(f"{channel} channel is saturated")
        elif check_underexposed(raw_data[:, i], threshold=1):
            print(f"{channel} channel is underexposed")
        else:
            print(f"{channel} channel is fine")
            usable_data[:, i] = raw_data[:, i]

    print("----------------------")

    plot_rgb_channels(usable_data, title="Usable data")

    # Remove the DC component
    usable_data = dc_block(usable_data, axis=0)
    plot_rgb_channels(usable_data, title="DC filtrert data")

    plt.tight_layout()
    plt.savefig("dc_blocked.png", dpi=300)

    # FFT of the raw data
    freq, fft_data = calculate_fft(usable_data, fs=args.fps, axis=0)
    psd_data = calculate_psd_from_fft(fft_data, fs=args.fps, axis=0)
    # Calculate the signal to noise ratio
    snr = calculate_snr_from_psd(
        freq[freq.shape[0] // 2 :], psd_data[psd_data.shape[0] // 2 :, :], axis=0
    )
    for i, channel in enumerate(["R", "G", "B"]):
        print(f"{channel} channel SNR: {snr[i]:.2f} dB")

    print("----------------------")

    plot_rgb_channels(
        np.abs(fft_data), x=freq, title="FFT", ylabel="Amplitude", xlabel="Frekvens (Hz)"
    )

    # If we assume a max human pulse of 200 bpm, we can calculate the frequency
    # of the pulse in Hz
    MAX_PULSE = 200
    MAX_PULSE_HZ = MAX_PULSE / 60
    print(f"Max pulse in Hz: {MAX_PULSE_HZ:.2f}")
    # We use the maximum pulse frequency as the cutoff frequency

    # Apply a lowpass filter to the data
    filtered_data = butter_filter(usable_data, cutoff=MAX_PULSE_HZ, fs=args.fps, axis=0)
    plot_rgb_channels(filtered_data, title="Lavpassfiltrert data")
    plt.tight_layout()
    plt.savefig("lowpass_filtered.png", dpi=300)

    # Apply a high pass filter to remove breathing artifacts
    # We assume that the human breathing is max 20 breaths per minute
    # And we assume that the heart rate is higher than 40 bpm
    MAX_BREATHING = 40
    MAX_BREATHING_HZ = MAX_BREATHING / 60
    print(f"Max breathing in Hz: {MAX_BREATHING_HZ:.2f}")
    # We use the maximum breathing frequency as the cutoff frequency
    filtered_data = butter_filter(
        filtered_data, cutoff=MAX_BREATHING_HZ, fs=args.fps, btype="high", axis=0
    )
    plot_rgb_channels(filtered_data, title="Høypassfiltrert data")
    plt.tight_layout()
    plt.savefig("highpass_filtered.png", dpi=300)

    # Apply windowing function to the data
    window = signal.windows.hann(filtered_data.shape[0])
    windowed_data = filtered_data * window[:, None]
    plot_rgb_channels(windowed_data, title="Data etter Hanning-vindu")
    plt.tight_layout()
    plt.savefig("windowed_data.png", dpi=300)

    # FFT of the data
    freq, fft_data = calculate_fft(filtered_data, fs=args.fps, axis=0)
    psd_data = calculate_psd_from_fft(fft_data, fs=args.fps, axis=0)

    plot_rgb_channels(
        np.abs(fft_data), x=freq, title="FFT", ylabel="Amplitude", xlabel="Frekvens (Hz)"
    )
    plt.tight_layout()
    plt.savefig("fft.png", dpi=300)
    plot_rgb_channels(
        np.abs(psd_data),
        x=freq,
        title="Normalisert effekttetthetsspektrum (PSD)",
        xlabel="Frekvens (Hz)",
        ylabel="Effekt ($V^2/$Hz)",
    )
    plt.tight_layout()
    plt.savefig("psd.png", dpi=300)

    freq_pos_half = freq[freq.shape[0] // 2 :]
    fft_pos_half_data = fft_data[fft_data.shape[0] // 2 :, :]
    psd_pos_half_data = psd_data[psd_data.shape[0] // 2 :, :]

    freq_pos_relevant = freq_pos_half[: freq_pos_half.shape[0] // 4]
    fft_pos_relevant_data = fft_pos_half_data[: fft_pos_half_data.shape[0] // 4, :]
    psd_pos_relevant_data = psd_pos_half_data[: psd_pos_half_data.shape[0] // 4, :]

    plot_rgb_channels(
        np.abs(fft_pos_relevant_data),
        x=freq_pos_relevant,
        title="FFT",
        ylabel="Amplitude",
        xlabel="Frekvens (Hz)",
    )
    plt.tight_layout()
    plt.savefig("fft.png", dpi=300)

    plot_rgb_channels(
        np.abs(psd_pos_relevant_data),
        x=freq_pos_relevant,
        title="Normalisert effekttetthetsspektrum (PSD)",
        xlabel="Frekvens (Hz)",
        ylabel="Effekt ($V^2/$Hz)",
    )
    plt.tight_layout()
    plt.savefig("psd_relevant.png", dpi=300)

    # Calculate the signal to noise ratio
    snr = calculate_snr_from_psd(freq_pos_relevant, psd_pos_relevant_data, axis=0)
    for i, channel in enumerate(["R", "G", "B"]):
        print(f"{channel} channel SNR: {snr[i]:.2f} dB")

    print("----------------------")

    # Caclulate the heart rate
    peak_freqs = freq_pos_relevant[np.argmax(psd_pos_relevant_data, axis=0)]
    heart_rate = peak_freqs * 60
    for i, channel in enumerate(["R", "G", "B"]):
        print(f"{channel} channel estimated heart rate: {heart_rate[i]:.2f} bpm")

    plt.show()
