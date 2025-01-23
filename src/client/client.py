"""This module contains the client class that will communicate with the server."""

import logging

import rpyc
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    conn = rpyc.connect("rpi32.local", 18861, config={"allow_pickle": True})
    sample_period, data = conn.root.run_adc_sample()

    # Make a local copy of the data
    local_data = np.array(data)
    local_sample_period = float(sample_period)
    print(f"Sample rate: {sample_period}")

    conn.close()

    print(f"Data: {local_data}")
    print(f"Data type: {type(local_data)}")
    print(f"Data shape: {local_data.shape}")
    local_data = local_data.T

    t = np.linspace(0, local_sample_period * local_data.shape[1], local_data.shape[1])
    t_ms = 1e3 * t
    fig, ax = plt.subplots(5, 1)
    for i, data_channel in enumerate(local_data):

        data_voltage = data_channel * (3.3 / (2**12 - 1))
        ax[i].plot(t_ms, data_voltage, label=f"Channel {i+1}")
        ax[i].set_xlabel("Time (ms)")
        ax[i].set_ylabel("Voltage (V)")
        ax[i].grid()
        ax[i].legend()

    # Do FFT of signals
    plt.figure()

    for i, data_channel in enumerate(local_data):
        data_voltage = data_channel * (3.3 / (2**12 - 1))

        freqs = np.fft.fftshift(np.fft.fftfreq(len(data_voltage), local_sample_period))
        fft = np.fft.fftshift(np.fft.fft(data_voltage))
        fft = np.abs(fft)
        plt.plot(freqs, fft, label=f"Channel {i+1}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.legend()

    plt.show()
