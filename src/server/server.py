"""This module contains a remote procedure call (RPC) service to read from ADCs."""

import subprocess
from subprocess import call
import logging
import pathlib
import time
import threading

from picamera2 import Picamera2  # pylint: disable=import-error
from picamera2.encoders import H264Encoder  # pylint: disable=import-error
import rpyc
from rpyc.utils.server import ThreadedServer
import numpy as np


logger = logging.getLogger(__name__)


class AdcSamplingService(rpyc.Service):
    """Class to implement the ADC sampling service.

    This service will provide the ability to read from ADCs connected to the Raspberry Pi.
    """

    def __init__(self):
        pass

    def on_connect(self, conn):
        # code that runs when a connection is created
        logger.info("Connection established")

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        logger.info("Connection closed")

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
    ) -> list[float, np.ndarray]:
        """Read from the ADCs and return the data.

        Returns:
            list[float, np.ndarray]: where the first argument is the samples.
            And the second is a list of floats representing the data read from the ADCs.
        """

        # Run src/adc_sampler to generate a binary file
        call(
            [
                pathlib.Path(__file__).parent / "../adc_sampler",
                str(sample_count),
                filename,
            ]
        )

        # Parse the binary file
        return self.__parse_binary(filename)


class CameraSamplingService(rpyc.Service):
    """Class to support making recordings using the Raspberry Pi camera."""

    camera: Picamera2

    h264_file = None
    mp4_file = None

    def __init__(self) -> None:
        pass

    def on_connect(self, conn):
        # code that runs when a connection is created
        logger.info("Connection established")
        self.camera = Picamera2()

        # Set default camera settings
        self.exposed_configure_camera()

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        logger.info("Connection closed")
        self.camera.close()
        del self.camera

    def exposed_configure_camera(
        self,
        iso: int = 10,
        frame_rate: int = 40,
        resolution: tuple[int, int] = (1640, 922),
        awb_gains: tuple[int, int] = (1, 2),
    ):
        """Configure the camera settings.

        Args:
            iso (int, optional): The ISO setting for the camera. Defaults to 10.
            frame_rate (int, optional): The frame rate for the camera. Defaults to 40.
            resolution (tuple[int,int], optional):
                The resolution for the camera. Defaults to (1640, 922).
        """
        self.camera.resolution = resolution
        self.camera.framerate = frame_rate

        # Set a low ISO. Adjusts the sensitivity of the camera. 0 means auto.
        # 10 most probably is the same
        # as 100, as picamera doc says it can only be set to 100, 200, 300, ..., 800.
        # Might probably want to set this to lowest possible value not equal to 0,
        # but can also be tuned.
        self.camera.iso = iso

        # Add a bit of delay here so that the camera has time to adjust its settings.
        # Skipping this will set a very low exposure and yield black frames, since the
        # camera needs to adjust the exposure to a high level according to current
        # light settings before turning off exposure compensation.
        logging.debug("Waiting for settings to adjust")
        time.sleep(2)

        # switch these two off so that we can manually control the awb_gains
        self.camera.exposure_mode = "off"
        self.camera.awb_mode = "off"

        # Set gain for red and blue channel.  Setting a single number (i.e.
        # camera.awb_gains = 1) sets the same gain for all channels, but 1 seems to be
        # too low for blue channel (frames constant black).  2 sometimes is not enough,
        # and is related to what happens during the sleep(2) above.  Probably has to be
        # tuned?
        self.camera.awb_gains = awb_gains

    def exposed_run_camera_sample(
        self, record_time: int = 30, filename: str = "foo"
    ) -> None:
        """Record a video using the Raspberry Pi camera.

        Args:
            filename (str, optional): The name of the file to save the video to.
                Defaults to "foo.h264".
        """

        h264_filename = filename + ".h264"
        mp4_filename = filename + ".mp4"

        # Start the preview
        self.camera.start_preview()
        time.sleep(5)

        # Start recording
        self.camera.start_recording(H264Encoder(), h264_filename)

        # Record for the amount of time we want
        time.sleep(record_time)

        # Stop recording
        self.camera.stop_recording()

        # Stop the preview
        self.camera.stop_preview()

        logging.info("Recording finished")

        subprocess.check_output(
            [
                "ffmpeg",
                "-framerate",
                str(self.camera.framerate),
                "-i",
                h264_filename,
                "-c",
                "copy",
                mp4_filename,
                "-y",  # Overwrite the file if it exists
            ]
        )

    def exposed_open_file(self, filename: str, mode: str = "rb") -> None:
        """Open the file in the default application.

        Args:
            filename (str): The name of the file to open.
        """

        h264_filename = filename + ".h264"
        mp4_filename = filename + ".mp4"

        self.h264_file = open(h264_filename, mode, encoding="utf-8")
        self.mp4_file = open(mp4_filename, mode, encoding="utf-8")

        return self.h264_file, self.mp4_file

    def exposed_close_file(self) -> None:
        """Close the files."""
        try:
            self.h264_file.close()
            self.mp4_file.close()
        except AttributeError:
            pass


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    connectionConfig = {"allow_public_attrs": True, "allow_pickle": True}

    adc_server = ThreadedServer(
        AdcSamplingService, port=18861, protocol_config=connectionConfig
    )
    camera_server = ThreadedServer(
        CameraSamplingService, port=18862, protocol_config=connectionConfig
    )

    # Start the servers
    t1 = threading.Thread(target=adc_server.start, name="ADC Server", daemon=True)
    t2 = threading.Thread(target=camera_server.start, name="Camera Server", daemon=True)

    t1.start()
    t2.start()

    # Have the main thread run until the user interrupts the program
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break
