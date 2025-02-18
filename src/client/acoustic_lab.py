"""This module contains methods for the acoustic lab."""

import logging

import rpyc
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

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

    plt.plot(mic1)
    plt.plot(mic2)
    plt.plot(mic3)
    plt.show()

    # Calculate the cross-correlation between the microphones
    corr_21 = np.abs(np.correlate(mic2, mic1, mode="full"))
    corr_31 = np.abs(np.correlate(mic3, mic1, mode="full"))
    corr_32 = np.abs(np.correlate(mic3, mic2, mode="full"))

    # Since we know that the maximum sample delay, we can ignore the rest of the cross-correlation
    detection_range = (
        len(corr_21) // 2 - max_sample_delay,
        len(corr_21) // 2 + max_sample_delay + 1,
    )
    corr_21 = corr_21[detection_range[0] : detection_range[1]]
    corr_31 = corr_31[detection_range[0] : detection_range[1]]
    corr_32 = corr_32[detection_range[0] : detection_range[1]]

    lags = signal.correlation_lags(len(mic1), len(mic2))[
        detection_range[0] : detection_range[1]
    ]
    print(lags)

    # Calculate the time delay between the microphones
    delay_21 = lags[np.argmax(corr_21)]
    delay_31 = lags[np.argmax(corr_31)]
    delay_32 = lags[np.argmax(corr_32)]

    logger.info("Delay 21: %d", delay_21)
    logger.info("Delay 31: %d", delay_31)
    logger.info("Delay 32: %d", delay_32)

    # Calculate the angle of arrival
    y = delay_31 + delay_21
    x = delay_31 - delay_21 + 2 * delay_32

    theta = np.arctan(-np.sqrt(3) * y / x)

    # If the x is negative, add 180 degrees
    # atan can only return values between -90 and 90 degrees
    if x < 0:
        theta += np.pi

    return np.degrees(theta)


def main(rpi_connection, real_time: bool = False, num_samples: int = 1000) -> None:
    """Main loop to perform continous acoustic lab measurements."""

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
                "acoustic_lab_data/sindre_135deg_1m_5.npz"
            ).values()

        # Estimate the angle of arrival
        theta = estimate_angle_of_arrival(data_voltage, 1 / sample_period)
        logger.info("Angle of arrival: %f", theta)


if __name__ == "__main__":

    # Connect to the Raspberry Pi
    pi_connection = rpyc.connect("rpi32.local", 18861, config={"allow_pickle": True})

    main(pi_connection, real_time=True)
