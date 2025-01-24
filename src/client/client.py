"""This module contains the client class that will communicate with the server."""

import logging

import rpyc
import numpy as np


import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

print("Using:", matplotlib.get_backend())

logger = logging.getLogger(__name__)


V_REF = 3.3  # Reference voltage for the ADC
ADC_RESOLUTION = 12  # ADC resolution in bits


def sample_value_to_voltage(value: int) -> float:
    """Convert an ADC sample value to a voltage."""
    return value * (V_REF / (2**ADC_RESOLUTION - 1))


def dc_block(data: np.ndarray) -> np.ndarray:
    """Remove DC offset from the data."""
    return data - np.mean(data)


if __name__ == "__main__":
    conn = rpyc.connect("rpi32.local", 18861, config={"allow_pickle": True})
    sample_period, data = conn.root.run_adc_sample()

    # Make a local copy of the data
    local_data = np.array(data)
    # Remove the first few samples as the start gives random values.
    local_data = local_data[10:]
    local_sample_period = float(sample_period)
    print(f"Sample rate: {sample_period}")

    conn.close()

    print(f"Data: {local_data}")
    print(f"Data type: {type(local_data)}")
    print(f"Data shape: {local_data.shape}")
    local_data = local_data.T

    print(f"Data: {local_data}")

    # Turn the data into voltages and remove DC offset.
    data_voltage = []
    for i, data_channel in enumerate(local_data):
        data_voltage.append(sample_value_to_voltage(data_channel))
        data_voltage[i] = dc_block(data_voltage[i])

    data_voltage = np.array(data_voltage)

    # Plot the data
    t_ms = (
        np.linspace(0, local_sample_period * local_data.shape[1], local_data.shape[1])
        * 1e3
    )
    fig, ax = plt.subplots(5, 1, tight_layout=True, sharex=True)
    for i, data_channel in enumerate(data_voltage):

        ax[i].plot(t_ms, data_channel, f"C{i}", label=f"Channel {i+1}")
        ax[i].set_ylabel("Voltage (V)")
        ax[i].grid()
        ax[i].legend()
        ax[i].set_ylim(-V_REF / 2, V_REF / 2)

    ax[-1].set_xlabel("Time (ms)")

    fig.suptitle("ADC Data")

    # Do FFT of signals
    plt.figure()

    for i, data_channel in enumerate(data_voltage):
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data_channel), local_sample_period))
        fft = np.fft.fftshift(np.fft.fft(data_channel))
        fft = np.abs(fft)
        plt.plot(freqs, fft, label=f"Channel {i+1}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.legend()

    plt.show()
