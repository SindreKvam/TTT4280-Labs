"""This module contains methods for the acoustic lab."""

import logging

import rpyc
import numpy as np
from scipy import signal

from common import sample_value_to_voltage, dc_block  # pylint: disable=import-error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SPEED_OF_SOUND = 343  # m/s
DISTANCE_BETWEEN_MICROPHONES = 0.055  # m


def estimate_angle_of_arrival(data: np.ndarray, sample_frequency: float) -> None:
    """Estimate the angle of arrival of the signal."""

    # Since the distance between the microphones is known, we can calculate the maximum time delay
    # between the microphones.
    max_sample_delay = int(
        (DISTANCE_BETWEEN_MICROPHONES / SPEED_OF_SOUND) * sample_frequency
    )

    # We have three microphones and two channels that are not used.
    mic2, mic3, mic1, _, _ = data

    # Calculate the cross-correlation between the microphones
    corr_21 = np.correlate(mic2, mic1, mode="full")
    corr_31 = np.correlate(mic3, mic1, mode="full")
    corr_32 = np.correlate(mic3, mic2, mode="full")

    lags = signal.correlation_lags(len(mic1), len(mic2))

    # Since we know that the maximum sample delay, we can ignore the rest of the cross-correlation
    detection_range = (
        len(corr_21) // 2 - max_sample_delay,
        len(corr_21) // 2 + max_sample_delay + 1,
    )
    corr_21 = corr_21[detection_range[0] : detection_range[1]]
    corr_31 = corr_31[detection_range[0] : detection_range[1]]
    corr_32 = corr_32[detection_range[0] : detection_range[1]]
    lags = lags[detection_range[0] : detection_range[1]]

    # Calculate the time delay between the microphones
    delay_21 = lags[np.argmax(corr_21)]
    delay_31 = lags[np.argmax(corr_31)]
    delay_32 = lags[np.argmax(corr_32)]

    logger.debug("Delay 21: %d", delay_21)
    logger.debug("Delay 31: %d", delay_31)
    logger.debug("Delay 32: %d", delay_32)

    # Calculate the angle of arrival
    y = np.sqrt(3) * delay_31 + delay_21
    x = delay_31 - delay_21 + 2 * delay_32

    theta = -np.arctan2(y, x)

    return np.degrees(theta)


def main(rpi_connection, real_time: bool = False, num_samples: int = 3000) -> None:
    """Main loop to perform continous acoustic lab measurements."""

    latest_theta_values = []
    while True:
        # Run the measurement
        data_voltage = []
        if real_time:
            # Fetch data directly from the Raspberry Pi
            sample_period, data = rpi_connection.root.run_adc_sample(num_samples)
            data = np.array(data).T

            for data_channel in data:
                data_voltage.append(dc_block(sample_value_to_voltage(data_channel)))

            data_voltage = np.array(data_voltage)
        else:
            # Load data from file
            sample_period, data_voltage = np.load(
                "acoustic_lab_data/1khz_200deg_30cm.npz"
            ).values()

        # Estimate the angle of arrival
        theta = estimate_angle_of_arrival(data_voltage, 1 / sample_period)
        logger.info("Angle of arrival: %f", theta)

        # Moving average and standard deviation of theta values
        latest_theta_values.append(theta)
        latest_theta_values = latest_theta_values[-10:]
        theta_avg = np.mean(latest_theta_values)
        theta_std = np.std(latest_theta_values, ddof=1)

        logger.info("Average angle of arrival: %f", theta_avg)
        logger.info("Standard deviation of angle of arrival: %f", theta_std)


if __name__ == "__main__":

    # Connect to the Raspberry Pi
    pi_connection = rpyc.connect("rpi32.local", 18861, config={"allow_pickle": True})

    main(pi_connection, real_time=True)
