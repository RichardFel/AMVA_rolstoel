# from proces_data import
from src.proces_data import proces_data
from src.prepare_data_2groups import prepare_data_2groups
from src.prepare_data_3groups import prepare_data_3groups
from src.predict_activities_2_groups import predict_activities_2groups
from src.predict_activities_3_groups import predict_activities_3groups
from src.predict_activities_ankle import predict_activities_3groups_ankle
from src.predict_activities_ankle_2groups import predict_activities_2groups_ankle
from src.predict_activities_arm import predict_activities_3groups_arm
from src.predict_activities_arm_2groups import predict_activities_2groups_arm

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import datetime
import matplotlib
matplotlib.use('TkAgg')


# Logging
today = datetime.date.today()
logging.basicConfig(filename=f'Logging/logging_{today}.log',
                    level=logging.WARNING, format='%(name)s - %(levelname)s - %(message)s')

settings = {
    'visualise': False,
    'data_path': 'Raw_data',
    'sensor_loc': 1,
    'Resample_freq': 10
}


def main():
    proces_data(settings)
    prepare_data_2groups()
    prepare_data_3groups()
    predict_activities_2groups()
    predict_activities_3groups()
    predict_activities_3groups_ankle()
    predict_activities_2groups_ankle()
    predict_activities_3groups_arm()
    predict_activities_2groups_arm()


if __name__ == "__main__":
    main()
