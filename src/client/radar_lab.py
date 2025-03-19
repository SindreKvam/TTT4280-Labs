"""This module contains methods for the radar lab."""

import logging
import argparse

import rpyc
import numpy as np
import matplotlib.pyplot as plt

from common import dc_block  # pylint: disable=import-error

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read data from the ADCs.")
    parser.add_argument(
        "--sample-time", type=float, default=1, help="How many seconds to sample for."
    )
    parser.add_argument(
        "--q-channel", type=int, default=4, help="Which channel to use for the Q channel."
    )
    parser.add_argument(
        "--i-channel", type=int, default=3, help="Which channel to use for the I channel."
    )
    parser.add_argument("--output", type=str, default="data.npz", help="Output file.")
    parser.add_argument(
        "--folder", type=str, default="radar_lab_data", help="Output folder."
    )
    args = parser.parse_args()

    # Connect to the Raspberry Pi
    pi_connection = rpyc.connect("rpi32.local", port=18861, config={"allow_pickle": True})

    # Run measurements
    sample_period, data = pi_connection.root.run_adc_sample(
        np.ceil(31250 * args.sample_time)
    )
    # Make local copies of the data
    sample_period = np.float64(sample_period)
    fs = 1 / sample_period
    data = np.array(data).T

    # Extract the IQ data
    data_IQ = np.array(data[args.i_channel] + 1j * data[args.q_channel])[10:]

    # Remove DC offset
    data_IQ = dc_block(data_IQ)

    # Save the data to file to have it for later.
    np.savez(f"{args.folder}/{args.output}", allow_pickle=True, data_IQ=data_IQ, fs=fs)

    # plt.plot(np.abs(data_IQ))
    plt.plot(data_IQ.real, label="$IF_I$")
    plt.plot(data_IQ.imag, label="$IF_Q$")

    plt.legend()
    plt.show()
