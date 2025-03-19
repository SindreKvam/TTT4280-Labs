"""Command line interface to run measurements related to radar lab."""

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Do analysis on the radar lab data.")
    parser.add_argument(
        "filename",
        type=str,
        default="radar_lab_data/data.npz",
        help="Path to the data file.",
    )
    args = parser.parse_args()

    # Load the data
    data = np.load(args.filename, allow_pickle=True)
    data_IQ = data["data_IQ"]
    fs = data["fs"]

    # Plot
    plt.plot(data_IQ.real, label="$IF_I$")
    plt.plot(data_IQ.imag, label="$IF_Q$")
    plt.xlabel("Sample number")
    plt.ylabel("Amplitude")
    plt.title("IQ Data")
    plt.legend()

    # Do analysis
    fft_data = np.fft.fftshift(np.fft.fft(data_IQ))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(data_IQ), 1 / fs))

    speed = freqs[np.argmax(fft_data)] * 3e8 / (2 * 24.13e9)
    print(f"Speed m/s: {speed:.2f}")
    print(f"Speed km/h: {speed * 3.6:.2f}")

    plt.figure()
    plt.plot(freqs, 20 * np.log10(np.abs(fft_data)))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("FFT of the data")
    plt.show()
