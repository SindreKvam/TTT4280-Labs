"""This module contains the client class that will communicate with the server."""

import logging
import time

from common import sample_value_to_voltage, dc_block, V_REF

import rpyc
import numpy as np
import dwfpy as dwf


logger = logging.getLogger(__name__)


def run_measurement_and_save_to_file(
    pi_connection, filename: str = "adc_data.npz"
) -> None:
    """Run a measurement on the Raspberry Pi and save the data to a file."""

    sample_period, data = pi_connection.root.run_adc_sample(31_260)

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

    # Save the data to a file that can be used for later analysis
    np.savez(
        f"output_data1/{filename}",
        sample_period=np.float64(local_sample_period),
        data=data_voltage,
    )


if __name__ == "__main__":
    conn = rpyc.connect("rpi32.local", 18861, config={"allow_pickle": True})

    run_measurement_and_save_to_file(conn)

    # Connect to the Analog Discovery 2
    with dwf.Device() as device:

        for f in np.logspace(1, 6, 100):

            print(f"Generating a {f:.2f}Hz sine wave on WaveGen channel 1...")
            device.analog_output["ch1"].setup(
                "sine", frequency=f, amplitude=V_REF / 2, offset=V_REF / 2, start=True
            )
            time.sleep(0.2)

            run_measurement_and_save_to_file(conn, filename=f"adc_data_{f:.2f}Hz.npz")

    conn.close()
