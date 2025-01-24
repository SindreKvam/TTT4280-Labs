"""This module contains the client class that will communicate with the server."""

from common import sample_value_to_voltage, dc_block, plot_data_channels

import logging

import rpyc
import numpy as np


import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

print("Using:", matplotlib.get_backend())

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    conn = rpyc.connect("rpi32.local", 18861, config={"allow_pickle": True})
    sample_period, data = conn.root.run_adc_sample(31_260)

    # Make a local copy of the data
    local_data = np.array(data)
    # Remove the first few samples as the start gives random values.
    local_data = local_data[10:]
    local_sample_period = float(sample_period)
    print(f"Sample period: {sample_period}")

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

    # Save the data to a file that can be used for later analysis
    np.savez(
        "output_data/adc_data.npz",
        sample_period=np.float64(local_sample_period),
        data=data_voltage,
    )

    # Often want to see what we have measured immediately.
    plot_data_channels(data_voltage, local_sample_period)
    plt.show()
