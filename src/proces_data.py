# %%
# Import modules
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import datetime
import matplotlib
matplotlib.use('TkAgg')


def resample_data(sensor_data, sample_freq, Resample_freq):
    removingNumber = int(
        np.floor(len(sensor_data) - (len(sensor_data)/sample_freq)*Resample_freq))
    interval = len(sensor_data)/removingNumber
    cumsumInterval = np.cumsum(np.ones(removingNumber)*interval) - 1
    removalInterval = [int(x) for x in cumsumInterval]
    return sensor_data.drop(removalInterval)


# %%
def proces_data(settings):
    file_details = pd.read_excel('Meta_data/file_details.xlsx')
    for file in os.listdir(settings['data_path']):

        # Skip different files
        if not file.endswith('.csv'):
            # continue

            subject, location = file.split('_')
            sensor, measurement, _ = location.split('.')

            # Check if ankle sensor
            if ((sensor == '1') | (sensor == '4') | (sensor == '5')):
                continue

            # Check if data already processed
            if os.path.isfile(f'Labelled_data/{subject}_{sensor}_{measurement}.csv'):
                continue

            # Print
            print(f'{subject}, sensor lacation {sensor}')

            # Load data
            sensor_data = pd.read_csv(f'{settings['data_path']}/{file}', names=['Time', 'Ax', 'Ay', 'Az',
                                                                                'Gx', 'Gy', 'Gz', 'True_time'], skiprows=9)
            time = sensor_data.pop('True_time')
            sensor_data = sensor_data.dropna()
            time = time.dropna()
            try:
                labeled_data = pd.read_csv(f'Labels/{subject}.{measurement}.csv',
                                           names=['Start', 'Eind', 'Duur', 'Activiteit',
                                                  'Label', 'Shake start', 'shake end'],
                                           usecols=[0, 1, 2, 3, 4, 5, 6],
                                           sep=';', skiprows=2)
            except pd.errors.ParserError:
                labeled_data = pd.read_csv(f'Labels/{subject}.{measurement}.csv',
                                           names=['Start', 'Eind', 'Duur', 'Activiteit',
                                                  'Label', 'Shake start', 'shake end'],
                                           usecols=[0, 1, 2, 3, 4, 5, 6],
                                           sep=';', skiprows=2)
            except FileNotFoundError:
                logging.warning(f'{file}: No labelled data')
                continue

            # Load labelled data
            try:
                # Shake start
                flip_start = labeled_data.dropna(
                    subset=['Shake start']).iloc[0]
                flip_time_start = datetime.datetime.strptime(
                    flip_start['Shake start'], '%H:%M:%S')
                flip_time_start = flip_time_start.hour * 3600 + \
                    flip_time_start.minute * 60 + flip_time_start.second
                start_label = datetime.datetime.strptime(
                    flip_start['Eind'], '%H:%M:%S')
                start_time = start_label.hour * 3600 + \
                    start_label.minute * 60 + start_label.second

                # Shake end
                flip_end = labeled_data.dropna(subset=['shake end']).iloc[0]
                flip_time_end = datetime.datetime.strptime(
                    flip_end['shake end'], '%H:%M:%S')
                flip_time_end = flip_time_end.hour * 3600 + \
                    flip_time_end.minute * 60 + flip_time_end.second
                end_label = datetime.datetime.strptime(
                    flip_end['Start'], '%H:%M:%S')
                end_time = end_label.hour * 3600 + \
                    end_label.minute * 60 + end_label.second
                labeled_data = labeled_data.dropna(subset=['Label'])
                expected_samples = (end_time - start_time) * \
                    settings['Resample_freq']
            except (ValueError, IndexError, TypeError):
                logging.warning(f'{file}: Label error')
                continue

            if len(sensor_data) < expected_samples:
                logging.warning(f'{file}: less samples than expected')
                continue

            fig, ax = plt.subplots()
            ax.plot(np.arange(0, 1800), sensor_data.iloc[:1800, 1:4])
            shake_1 = plt.ginput(1)

            fig, ax = plt.subplots()
            ax.plot(np.arange(0, 1800), sensor_data.iloc[-1801:-1, 1:4])
            shake_2 = plt.ginput(1)

            # Resample sensor data
            new_sensor_data = sensor_data.iloc[int(
                shake_1[0][0]):len(sensor_data) - int(shake_2[0][0]) + 1800, :]
            new_sensor_data = new_sensor_data.reset_index()
            measurement_duration = flip_time_end - flip_time_start
            sample_freq = len(new_sensor_data) / measurement_duration
            new_sensor_data = resample_data(
                new_sensor_data, sample_freq, settings['Resample_freq'])

            # correct for the seconds difference
            new_start = (start_time - flip_time_start) * 10
            new_end = (flip_time_end - end_time) * 10
            if new_end == 0:
                new_sensor_data = new_sensor_data.iloc[new_start:]
            else:
                new_sensor_data = new_sensor_data.iloc[new_start:-new_end]

            # Labels
            start = labeled_data.loc[labeled_data['Start']
                                     == flip_start['Eind']].index[0]
            end = labeled_data.loc[labeled_data['Eind']
                                   == flip_end['Start']].index[0]
            labels = labeled_data.loc[start:end]

            tmp = np.zeros(len(new_sensor_data))
            for idx, label in enumerate(labels['Label']):
                tmp[idx*50:idx*50+50] = label

            # Add labels to data
            labeled_sensor_data = new_sensor_data.reset_index(
                drop=True).join(pd.DataFrame(tmp, columns=['Label']))

            if settings['visualise']:
                fig, ax = plt.subplots(2)
                ax[0].plot(labeled_sensor_data.loc[((labeled_sensor_data['Label'] >= 5) & (
                    labeled_sensor_data['Label'] <= 8)), 'Ax':'Az'].reset_index(drop=True))
                ax[0].set_title('Active')
                ax[1].plot(labeled_sensor_data.loc[((labeled_sensor_data['Label'] >= 2) & (
                    labeled_sensor_data['Label'] <= 3)), 'Ax':'Az'].reset_index(drop=True))
                ax[1].set_title('Inactive')
                plt.tight_layout()
                plt.show()

            # Save data
            labeled_sensor_data.to_csv(
                f'Labelled_data/{subject}_{sensor}_{measurement}.csv')

            # Save metadata
            file_details.loc[f'{subject}_{sensor}_{measurement}'] = [subject, sensor, measurement, flip_time_start,
                                                                     start_time, flip_time_end, end_time,
                                                                     expected_samples, int(shake_1[0][0]), int(shake_2[0][0])]
            file_details.to_excel('Meta_data/file_details.xlsx', index=0)

            # Close plot
            plt.close('all')

# %%
