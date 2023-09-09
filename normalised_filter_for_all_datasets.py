import pandas as pd
from matplotlib import pyplot as plt
import scipy.signal
import numpy as np
import snr_measure
import kalman_filter_final
import list_for_table


class plot_normalised:

    def __init__(self, file, title_of_graph):
        self.file = file
        self.title1 = title_of_graph

    snr_value = 0
    snr_kalman = 0
    lst = [('DATASET', 'FILTER', 'SNR ratio (dB)')]

    @staticmethod
    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def working(self):
        columns = ["timestamp", "data"]
        # df = pd.read_csv("rundatanew.csv", usecols=columns)
        df = pd.read_csv(self.file, usecols=columns)
        print(type(df))
        df1 = df[1:1000]
        # sampleRate, data = scipy.io.wavfile.read('test2.wav')
        # the length of data should be changed accordingly
        sampleRate, data = 1000, df1.data
        times = np.arange(len(data)) / sampleRate

        d_ecg, filtered = self.filter_data(df1)

        # applying normalisation to the derived raw data
        normalized_raw_data = self.NormalizeData(d_ecg)

        normalized_filter = self.NormalizeData(filtered)
        return normalized_raw_data, normalized_filter

        # normalization is applied to the filtered data

    def filter_data(self, df1):
        # applying a 3-pole lowpass filter at 0.1x Nyquist frequency
        b, a = scipy.signal.butter(3, 0.1, 'low', analog=False)
        # d_ecg, peaks_d_ecg = l2f.decg_peaks(df1.data, time=times)
        d_ecg = np.diff(df1.data)  # find derivative of ecg signal
        filtered = scipy.signal.filtfilt(b, a, d_ecg)
        filtered = np.round(filtered, 2)
        return d_ecg, filtered

    def plot_graph(self):
        normalized_raw_data, normalized_filter = self.working()
        plt.figure(self.title1, (19, 10))
        plt.subplot(121)
        plt.plot(normalized_raw_data)
        plt.title("Normalized raw data")
        plt.margins(0, .05)

        plt.subplot(122)
        plt.plot(normalized_filter)
        plt.title("Normalised filtered data")
        plt.margins(0, .05)

        plt.tight_layout()
        plt.show()

    def plot_kalman(self):
        dt = 1.0 / 60
        F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
        H = np.array([1, 0, 0]).reshape(1, 3)
        Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
        R = np.array([0.5]).reshape(1, 1)

        normalized_raw_data, normalized_filter = self.working()
        measurements = normalized_raw_data

        kf = kalman_filter_final.KalmanFilter(F=F, H=H, Q=Q, R=R)
        predictions = []  # this will be considered as filtered data (normalised)

        for z in measurements:
            predictions.append(np.dot(H, kf.predict())[0])
            kf.update(z)

        plt.figure(self.title1, (19, 10))
        plt.plot(range(len(measurements)), measurements, label='Normalized raw data')
        plt.plot(range(len(predictions)), np.array(predictions), label='Normalized Kalman Filtered Data')
        plt.legend()
        plt.show()
        normalized_raw_data, normalized_filter = self.working()
        result1 = snr_measure.snr_value(normalized_raw_data, np.array(predictions))
        result2 = result1.measure_snr()
        result2 = np.round(result2, 4)
        plot_normalised.snr_kalman = result2  # snr_kalman is a class variable
        print(f'SNR value for {self.title1} with Kalman filter: {result2} dB')
        str2 = 'Kalman filter'
        title_ = ""
        snr_ = str(plot_normalised.snr_kalman)
        lst1 = [title_, str2, snr_]
        plot_normalised.lst.append(tuple(lst1))

    def measure_snr_all(self):  # for butterworth filter
        normalized_raw_data, normalized_filter = self.working()
        result1 = snr_measure.snr_value(normalized_raw_data, normalized_filter)
        result2 = result1.measure_snr()
        result2 = np.round(result2, 4)
        plot_normalised.snr_value = result2
        print(f'SNR value for {self.title1} with Butterworth filter: {result2} dB')

    def tabular_data(self):
        str1 = 'Butterworth filter'
        title_ = str(self.title1)
        snr_ = str(plot_normalised.snr_value)
        lst1 = [title_, str1, snr_]
        plot_normalised.lst.append(tuple(lst1))

    @staticmethod
    def print_table_data():
        table_data_object1 = list_for_table.list_table(tuple(plot_normalised.lst))
        table_data_object1.add_data()
        table_data_object1.print_table()


walking_data = plot_normalised("WALKING_DATA.csv", "WALKING DATA")
walking_data.plot_graph()
walking_data.measure_snr_all()
walking_data.tabular_data()
walking_data.plot_kalman()
running_data = plot_normalised("rundatanew.csv", "RUNNING DATA")
running_data.plot_graph()
running_data.measure_snr_all()
running_data.tabular_data()
running_data.plot_kalman()
resting_data = plot_normalised("test2.csv", "RESTING DATA")
resting_data.plot_graph()
resting_data.measure_snr_all()
resting_data.tabular_data()
resting_data.plot_kalman()
resting_data.print_table_data()

# walking_psnr = psnr_trial.PSNR("walking_normalised_raw.png", "walking_normalised_filtered.png")
# result_walking = walking_psnr.result()
#
# running_psnr = psnr_trial.PSNR("running_normalised_raw.png", "running_normalised_filtered.png")
# result_running = running_psnr.result()
#
# resting_psnr = psnr_trial.PSNR("normalised_raw_data_image.png", "normalised_filtered_data_image.png")
# result_resting = resting_psnr.result()
