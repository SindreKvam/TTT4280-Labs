"""This module contains common functions used by the client."""

import logging
import numpy as np
import matplotlib

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


def plot_fft_channels(data: np.ndarray, sample_period: float) -> None:
    """Plot the FFT of the data from the ADCs."""

    f = np.fft.fftshift(np.fft.fftfreq(data.shape[1], sample_period))

    fig, ax = plt.subplots(5, 1, tight_layout=True, sharex=True)
    for i, data_channel in enumerate(data):

        data_channel_fft = np.fft.fftshift(np.fft.fft(data_channel))
        ax[i].plot(f, np.abs(data_channel_fft), f"C{i}", label=f"Channel {i+1}")
        ax[i].set_ylabel("Magnitude")
        ax[i].grid()
        ax[i].legend()
        ax[i].set_xlim(-2e3, 2e3)

    ax[-1].set_xlabel("Frequency (Hz)")

    fig.suptitle("ADC Data FFT")
