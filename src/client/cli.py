"""Command line interface for the client module."""

from common import (
    plt,
    plot_data_channels,
    plot_fft_channels,
    calculate_power,
    listen_to_signal,
)

import argparse
import logging

import numpy as np

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read data from the ADCs.")
    parser.add_argument(
        "-f",
        "--signal-filename",
        type=str,
        help="The filename to read the signal data from.",
        default="output_data/adc_data.npz",
    )
    parser.add_argument(
        "-n",
        "--noise-filename",
        type=str,
        help="The filename to read the noise data from.",
        default="",  # No noise data by default
    )
    parser.add_argument(
        "-p",
        "--plot",
        action=argparse.BooleanOptionalAction,
        help="Plot the data.",
        default=True,
    )
    parser.add_argument(
        "--snr",
        action=argparse.BooleanOptionalAction,
        help="Calculate the SNR.",
        default=False,
    )
    parser.add_argument(
        "--listen",
        action=argparse.BooleanOptionalAction,
        help="Listen to the signal.",
        default=False,
    )
    args = parser.parse_args()

    sample_period, data = np.load(args.signal_filename).values()

    sample_rate = 1 / sample_period
    print(f"Sample period: {sample_period}")
    print(f"Sample rate: {sample_rate}")
    print(f"Data: {data}")

    if args.plot:
        plot_data_channels(data, sample_period)
        plot_fft_channels(data, sample_period)

    if args.listen:
        listen_to_signal(data, sample_period)

    snr = np.zeros(data.shape[0])
    if args.snr:
        if not args.noise_filename:
            raise ValueError("No noise file provided.")

        noise_data = np.load(args.noise_filename)["data"]

        for i in range(data.shape[0]):
            snr[i] = calculate_power(data[i]) / calculate_power(noise_data[i])

        snr_db = 10 * np.log10(snr)

        print(f"SNR (dB): {snr_db}")

    plt.show()
