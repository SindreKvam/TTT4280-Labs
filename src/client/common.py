"""This module contains common functions used by the client."""

import logging
import numpy as np
import matplotlib
from scipy import signal

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import sounddevice as sd

V_REF = 3.3  # Reference voltage for the ADC
ADC_RESOLUTION = 12  # ADC resolution in bits

logger = logging.getLogger(__name__)


def sample_value_to_voltage(value: int) -> float:
    """Convert an ADC sample value to a voltage."""
    return value * (V_REF / (2**ADC_RESOLUTION))


def dc_block(data: np.ndarray) -> np.ndarray:
    """Remove DC offset from the data."""
    return data - np.mean(data)


def calculate_power(data: np.ndarray, R_L: float = 1.0) -> float:
    """Calculate the power of the signal."""
    return np.mean(np.pow(np.abs(data), 2)) / R_L


def plot_data_channels(data: np.ndarray, sample_period: float) -> None:
    """Plot the data from the ADCs."""

    t_ms = np.linspace(0, sample_period * data.shape[1], data.shape[1]) * 1e3

    fig, ax = plt.subplots(5, 1, tight_layout=True, sharex=True)
    for i, data_channel in enumerate(data):

        ax[i].plot(t_ms, data_channel, f"C{i}", label=f"Channel {i+1}")
        ax[i].grid()
        ax[i].legend()
        ax[i].set_ylim(-V_REF / 2 - V_REF * 0.1, V_REF / 2 + V_REF * 0.1)

    fig.supxlabel("Time (ms)")
    fig.supylabel("Voltage (V)", rotation="vertical")

    fig.suptitle("ADC Data")


def listen_to_signal(data: np.ndarray, sample_period: float) -> None:
    """Listen to the signal from the ADCs."""

    t_ms = np.linspace(0, sample_period * data.shape[1], data.shape[1]) * 1e3

    fig, ax = plt.subplots(5, 1, tight_layout=True, sharex=True)
    for i, data_channel in enumerate(data):

        sd.play(data_channel, int(1 / sample_period))
        sd.wait()

        ax[i].plot(t_ms, data_channel, f"C{i}", label=f"Channel {i+1}")
        ax[i].grid()
        ax[i].legend()
        ax[i].set_ylim(-V_REF / 2 - V_REF * 0.1, V_REF / 2 + V_REF * 0.1)

    fig.supxlabel("Time (ms)")
    fig.supylabel("Voltage (V)", rotation="vertical")

    fig.suptitle("ADC Data")


def generate_psd(
    data: np.ndarray,
    sample_period: float,
    *,
    nfft: int = 31_250,
    window_func=signal.windows.boxcar,
) -> None:
    """"""

    # Add window function
    data = data * window_func(len(data))

    # FFT of the data
    data_fft = np.fft.fft(data, n=nfft, norm=None)
    data_fft = np.abs(data_fft[: len(data_fft) // 2])
    data_psd = np.abs(data_fft * data_fft)
    psd_normalized = np.abs(data_psd / np.sum(data_psd))

    f = np.linspace(0, 1 / (2 * sample_period), len(data_psd))

    return f, psd_normalized


def plot_fft_channels(data: np.ndarray, sample_period: float) -> None:
    """Plot the FFT of the data from the ADCs."""

    fig, ax = plt.subplots(5, 1, tight_layout=True, sharex=True)
    for i, data_channel in enumerate(data):

        # FFT of the data
        ax[i].plot(
            *generate_psd(
                data_channel, sample_period, nfft=31250, window_func=signal.windows.boxcar
            ),
            label=f"Channel {i+1}",
        )

        # Zero padding up to 2^17
        ax[i].plot(
            *generate_psd(
                data_channel, sample_period, nfft=2**17, window_func=signal.windows.boxcar
            ),
            label=f"Channel {i+1} zero-padded",
        )

        # Window function of the data
        ax[i].plot(
            *generate_psd(
                data_channel, sample_period, nfft=31250, window_func=signal.windows.hann
            ),
            label=f"Channel {i+1} windowed",
        )

        # Window function of the data and zero padding
        ax[i].plot(
            *generate_psd(
                data_channel, sample_period, nfft=2**17, window_func=signal.windows.hann
            ),
            label=f"Channel {i+1} windowed and zero padded",
        )

        ax[i].set_ylabel("Normalized Magnitude")
        ax[i].grid()
        ax[i].legend()
        ax[i].set_xlim(990, 1010)

    ax[-1].set_xlabel("Frequency (Hz)")

    fig.suptitle("ADC Data FFT")
