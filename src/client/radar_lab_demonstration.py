"""This module contains hard-coded methods to demonstrate the radar lab."""

import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

# Negativ ting ved å mikse signalet ned til 0 Hz.
# legger til 1/f støy på målingene.


# 2. Vis at alle hastighetsmålingene har blitt gjort (minst 12).


# 4. Estimat av SNR basert på frekvensspektrumet

# 5. Beregn varians/standardavvik av hastighetsmålingene


def calculate_speed_from_doppler_shift(
    *, doppler_frequency: float, radar_frequency: float
) -> float:
    """Calculate the speed from the doppler shift"""
    return doppler_frequency * 3e8 / (2 * radar_frequency)


if __name__ == "__main__":

    # 1. Bode plott av begge båndpassfiltrene
    fig, ax = plt.subplots(2, 1, tight_layout=True, sharex=True)
    ax2 = [0, 0]
    for index, path in enumerate(Path("radar_lab_data").rglob("*bandpass*.csv")):
        print(path, path.name)
        dataframe = pd.read_csv(path, comment="#")

        ax2[index] = ax[index].twinx()
        dataframe.plot(
            x="Frequency (Hz)",
            y=["Channel 1 Magnitude (dB)", "Channel 2 Magnitude (dB)"],
            label=["Magnitude 1 [dB]", "Magnitude 2 [dB]"],
            xlabel="Frequency (Hz)",
            ylabel="Magnitude (dB)",
            ax=ax[index],
            grid=True,
            logx=True,
        )

        dataframe.plot(
            x="Frequency (Hz)",
            y="Channel 2 Phase (deg)",
            label="Phase [deg]",
            xlabel="Frequency (Hz)",
            ylabel="Phase (deg)",
            ax=ax2[index],
            logx=True,
            grid=True,
            alpha=0.8,
            color="C3",
        )
        plt.title(path.name)

    # 2. Vis at alle hastighetsmålingene har blitt gjort (minst 12).
    all_speeds_kmh = [[], [], []]
    speed_index = (
        0  # There are three different speeds measured, split them into different lists
    )
    for index, path in enumerate(Path("radar_lab_data").rglob("laptop*.npz"), start=1):
        print("----------------------")

        actual_speed_kmh = float(re.findall(r"laptop_(.+)m", path.name)[0])
        print(f"Actual speed: {actual_speed_kmh} km/h")

        print(path.name)
        data = np.load(path, allow_pickle=True)

        # Remove 10 first samples, as they are invalid
        data_IQ = np.array(data["data_IQ"])[10:]
        fs = np.float64(data["fs"])

        # Calculate the complex FFT of the data
        fft_data = np.fft.fftshift(np.fft.fft(data_IQ))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(data_IQ), 1 / fs))
        fft_data_db = 20 * np.log10(np.abs(fft_data))
        fft_data_db_normalized = fft_data_db - np.max(fft_data_db)

        speed = calculate_speed_from_doppler_shift(
            doppler_frequency=freqs[np.argmax(fft_data)], radar_frequency=24.13e9
        )
        speed_kmh = speed * 3.6
        print(f"Speed m/s: {speed:.2f}")
        print(f"Speed km/h: {speed_kmh:.2f}")

        # 4. Estimat av SNR basert på frekvensspektrumet
        _peak_frequencies = freqs[np.argmax(fft_data)]
        FREQ_MARGIN = 5  # Hz
        lower_bounds = _peak_frequencies - FREQ_MARGIN  # Hz
        upper_bounds = _peak_frequencies + FREQ_MARGIN  # Hz
        _signal_mask = (freqs >= lower_bounds) & (freqs <= upper_bounds)
        print(_signal_mask)

        psd = np.abs(fft_data) ** 2 / (fft_data.shape[0] * fs)

        _delta_freq = freqs[1] - freqs[0]
        _signal = psd * _signal_mask
        _noise = psd * ~_signal_mask
        _signal_total = np.sum(_signal) * _delta_freq
        _noise_total = np.sum(_noise) * _delta_freq
        snr = 10 * np.log10(_signal_total / _noise_total)
        print(f"SNR: {snr:.2f} dB")

        # 3. Plott av tidsserrie og spektrum av minst en hastighetsmåling
        if index in (1, 5, 9):

            plt.figure()
            plt.plot(data_IQ.real, label="$IF_I$")
            plt.plot(data_IQ.imag, label="$IF_Q$")
            plt.title("IQ Data")
            plt.legend()

            plt.figure()
            plt.plot(freqs, fft_data_db_normalized)
            plt.xlim(-2.3e3, 2.3e3)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Magnitude")
            plt.title("FFT of the data")

            plt.figure()
            plt.plot(freqs, _signal, label="Signal")
            plt.plot(freqs, _noise, label="Noise")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Magnitude")
            plt.title("Signal and noise of PSD")
            plt.legend()

        # 5. Beregn varians/standardavvik av hastighetsmålingene
        all_speeds_kmh[speed_index].append(speed_kmh)

        if index % 4 == 0:
            speed_index += 1

    print("----------------------")
    print(np.array(all_speeds_kmh))
    all_speeds_kmh = np.array(all_speeds_kmh)
    std = np.std(all_speeds_kmh, axis=1)
    mean = np.mean(all_speeds_kmh, axis=1)
    print(f"Mean: {mean}")
    print(f"Standard deviation: {std}")

    plt.show()
