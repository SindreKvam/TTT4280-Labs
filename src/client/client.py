"""This module contains the client class that will communicate with the server."""

import logging

import rpyc
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    conn = rpyc.connect("rpi32.local", 18861, config={"allow_pickle": True})
    sample_rate, data = conn.root.run_adc_sample()

    # Make a local copy of the data
    local_data = np.array(data)
    print(f"Sample rate: {sample_rate}")

    conn.close()

    print(f"Data: {local_data}")
    print(f"Data type: {type(local_data)}")
    print(f"Data shape: {local_data.shape}")
    local_data = local_data.T
    for i, channel in enumerate(local_data):
        plt.plot(channel, label=f"Channel {i}")

    plt.grid()
    plt.legend()
    plt.show()
