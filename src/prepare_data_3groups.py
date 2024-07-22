import pandas as pd
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt


def prepare_data_3groups():
    # Settings
    data_path = 'Labelled_data'

    zitten = [2, 3, 4, 9, 14, 15, 16, 17, 24]
    lopen = [5, 6, 7, 8, 18, 19, 22, 26, 10, 12, 13, 33, 23]
    rolstoel = [25, 27, 29, 30, 31]

    # functions

    def def_categories(x):
        if x in zitten:
            return 0
        elif x in lopen:
            return 1
        elif x in rolstoel:
            return 2

    value_frequency_n_walking = pd.Series(np.zeros(35), np.arange(35))
    value_frequency_walking = pd.Series(np.zeros(35), np.arange(35))
    value_frequency_categories = pd.Series(np.zeros(3), np.arange(3))
    charact = pd.read_excel(
        'Characteristics/Participantkarakteristieken_Rolstoel_v1.xlsx', sheet_name='Opgeschoond')
    nWalking = charact.loc[(charact['Onderzoeksgroep'] == 4),
                           'Proefpersoonnummer'].unique()

    for file in os.listdir(data_path):
        if file.startswith('.'):
            continue
        subj = file.split('_')[0]

        sensor_data = pd.read_csv(f'{data_path}/{file}', index_col=0)

        if int(subj) in nWalking:
            value_frequency_n_walking = value_frequency_n_walking.add(
                sensor_data['Label'].value_counts() / 50, fill_value=0)
        else:
            value_frequency_walking = value_frequency_walking.add(
                sensor_data['Label'].value_counts() / 50, fill_value=0)

        # Set labels to labels ML
        sensor_data['Categories'] = sensor_data['Label'].apply(def_categories)

        # Drop unused sections of data
        sensor_data = sensor_data.dropna(subset=['Categories'])
        sensor_data = sensor_data.drop(columns=['Time', 'Label'])

        value_frequency_categories = value_frequency_categories.add(
            sensor_data['Categories'].value_counts() / 50, fill_value=0)

        # Save data to processed_data
        sensor_data.to_csv(f'Processed_data/{file}')
