"""This module contains the client class that will communicate with the server."""

import logging
import time
import argparse
from common import (
    sample_value_to_voltage,
    dc_block,
    V_REF,
)  # pylint: disable=import-error

import rpyc
import numpy as np
import dwfpy as dwf


logger = logging.getLogger(__name__)


def run_measurement(pi_connection, num_samples: int = 31_260) -> None:
    """Run a measurement on the Raspberry Pi and save the data to a file."""

    sample_period, data = pi_connection.root.run_adc_sample(num_samples)

    # Make a local copy of the data
    local_data = np.array(data)
    # Remove the first few samples as the start gives random values.
    local_data = local_data[10:]
    local_sample_period = float(sample_period)
    logger.debug(f"Sample period: {sample_period}")

    logger.debug(f"Data: {local_data}")
    logger.debug(f"Data type: {type(local_data)}")
    logger.debug(f"Data shape: {local_data.shape}")
    local_data = local_data.T

    logger.debug(f"Data: {local_data}")

    # Turn the data into voltages and remove DC offset.
    data_voltage = []
    for i, data_channel in enumerate(local_data):
        data_voltage.append(sample_value_to_voltage(data_channel))
        data_voltage[i] = dc_block(data_voltage[i])

    data_voltage = np.array(data_voltage)

    return data_voltage, local_sample_period


def save_data_to_file(
    data: np.ndarray,
    sample_period: float,
    *,
    filename: str = "adc_data.npz",
    output_dir: str = "output_data",
) -> None:
    """Save the data to a file that can be used for later analysis."""
    np.savez(
        f"{output_dir}/{filename}",
        sample_period=np.float64(sample_period),
        data=data,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a measurement on the Raspberry Pi.")
    parser.add_argument(
        "-i",
        "--ip",
        type=str,
        help="The IP address of the Raspberry Pi.",
        default="rpi32.local",
    )
    parser.add_argument(
        "-s", "--frequency_sweep", action="store_true", help="Run a frequency sweep."
    )
    parser.add_argument(
        "-o", "--output", type=str, help="The output filename.", default="adc_data.npz"
    )
    parser.add_argument(
        "--output-dir", type=str, help="The output directory.", default="output_data"
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        help="The number of samples to collect.",
        default=31_260,
    )
    args = parser.parse_args()

    # Connect to the instrument
    conn = rpyc.connect(args.ip, 18861, config={"allow_pickle": True})

    # Run measurements
    data, sample_period = run_measurement(conn, num_samples=args.num_samples)
    save_data_to_file(
        data, sample_period, filename=args.output, output_dir=args.output_dir
    )

    if args.frequency_sweep:
        # Connect to the Analog Discovery 2 and run frequency sweep
        with dwf.Device() as device:

            for f in np.logspace(1, 6, 100):

                print(f"Generating a {f:.2f}Hz sine wave on WaveGen channel 1...")
                device.analog_output["ch1"].setup(
                    "sine", frequency=f, amplitude=V_REF / 2, offset=V_REF / 2, start=True
                )
                time.sleep(0.2)

                data, sample_period = run_measurement(conn)
                save_data_to_file(data, sample_period, filename=f"adc_data_{f:.2f}Hz.npz")

    conn.close()
