"""This module contains a remote procedure call (RPC) service to read from ADCs."""

from subprocess import call

import rpyc
from rpyc.utils.server import ThreadedServer
import numpy as np


class AdcSamplingService(rpyc.Service):
    """Class to implement the ADC sampling service.

    This service will provide the ability to read from ADCs connected to the Raspberry Pi.
    """

    def __init__(self):
        pass

    def on_connect(self, conn):
        # code that runs when a connection is created
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        pass

    def __parse_binary(self, filename: str, channels: int = 5) -> list[float]:
        """
        Parse data produced using adc_sampler.c.

        Returns sample period and a (`samples`, `channels`) `float64` array of
        sampled data from all `channels` channels.
        """

        with open(filename, "r", encoding="utf-8") as fid:
            sample_period = np.fromfile(fid, count=1, dtype=float)[0]
            data = np.fromfile(fid, dtype="uint16").astype("float64")
            # The "dangling" `.astype('float64')` casts data to double precision
            # Stops noisy autocorrelation due to overflow
            data = data.reshape((-1, channels))

        # sample period is given in microseconds, so this changes units to seconds
        sample_period *= 1e-6
        return sample_period, data

    def exposed_run_adc_sample(
        self, sample_count: int = 1024, filename: str = "foo.bin"
    ) -> list[float]:
        """Read from the ADCs and return the data.

        Returns:
            list[float]: where the first argument is the samples.
            And the second is a list of floats representing the data read from the ADCs.
        """

        # Run src/adc_sampler to generate a binary file
        call(["../adc_sampler", sample_count, filename])

        # Parse the binary file
        return self.__parse_binary(filename)


if __name__ == "__main__":
    t = ThreadedServer(AdcSamplingService, port=18861)
    t.start()
