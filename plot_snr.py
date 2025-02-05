"""Plot signal to noise ratio over multiple frequencies."""

from pathlib import Path
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {"family": "normal", "size": 12}
matplotlib.rc("font", **font)

from src.client.common import calculate_power

signal_power = np.zeros((5, 100))
noise_power = np.ones((5, 100))
freq_array = np.zeros((5, 100))
sample_frequencies = np.zeros(100)

index = 0

pathlist = Path("snr_data").glob("**/*.npz")
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    print(path_in_str)

    file_data = np.load(path_in_str)
    sample_period = file_data["sample_period"]
    sample_freq = 1 / sample_period
    data = file_data["data"]

    if "noise" in path_in_str:
        # print("Noise")
        for i, val in enumerate(data):
            noise_power[i] *= calculate_power(val)
            # print(noise_power)
            # print(data[i], data[i].shape)
            # print(calculate_power(data[i]))

    else:
        # print("Signal")
        for i, val in enumerate(data):
            freq = float(re.findall(r"adc_data_(.*)Hz", path_in_str)[0])

            # if freq > 1e5:
            #     continue

            signal_power[i][index] = calculate_power(val)
            freq_array[i][index] = freq
            sample_frequencies[index] = 1 / sample_period

        index += 1

    file_data = np.load(path_in_str)

    # print(data.shape)

SNR_MAX = 80

snr = np.zeros((5, 100))
for channel in range(5):
    snr[channel] = 10 * np.log10(signal_power[channel] / noise_power[channel])
    snr[channel][snr[channel] > SNR_MAX] = SNR_MAX

print(freq_array)
for channel in range(5):
    plt.scatter(freq_array[channel], snr[channel], label=f"Channel {channel+1}")

sample_freq = 1 / sample_frequencies.shape[0] * np.sum(sample_frequencies)

plt.axvline(sample_freq / 2, color="red", linestyle="--", label="Nyquist Frequency")
plt.axvspan(sample_freq / 2, 1e6, color="red", alpha=0.1)

plt.axhline(74, color="black", linestyle="--", label="Theoretical max SNR")
plt.axhspan(74, SNR_MAX, color="black", alpha=0.1)

plt.xscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("SNR (dB)")
plt.legend(prop={"size": 10})
plt.grid()
plt.title("Signal to Noise Ratio")
plt.tight_layout()
plt.savefig("signal_to_noise_ratio.png")

plt.show()
