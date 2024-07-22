import numpy as np
# Functions


def z_score(arr):
    # Calculate z_scores
    return (arr - arr.mean()) / arr.std()


def detect_outliers(sensor_data):
    spread_outlier = 4

    # Initialize an array to store the outlier indexes
    outliers = np.array([])

    # Loop through each dimension of the sensor data
    for dim in range(sensor_data.shape[2]):
        # Calculate the mean and standard deviation along axis 1 (across time) for the current dimension
        avr = sensor_data[:, :, dim].mean(axis=1)
        std = sensor_data[:, :, dim].std(axis=1)

        # Calculate the z-scores for both mean and standard deviation values
        z_avr = z_score(avr)
        z_std = z_score(std)

        # Concatenate the indexes where z-scores for mean and standard deviation exceed 5
        outliers = np.concatenate(
            (outliers, (np.where(z_avr > spread_outlier)[0])))
        outliers = np.concatenate(
            (outliers, (np.where(z_std > spread_outlier)[0])))

    # Return unique outlier indexes
    return np.unique(outliers)


def remove_outliers(category_data,  category_part):
    # Detect outliers using the provided function
    outlier_sit = detect_outliers(category_data).astype(int)

    # Remove outliers from the data arrays
    data_clean = np.delete(category_data, outlier_sit, axis=0)
    part_clean = np.delete(category_part, outlier_sit, axis=0)

    # Calculate the number of outliers removed
    num_removed = len(category_data) - len(data_clean)
    print(f'Outliers removed: {num_removed}')

    return data_clean, part_clean
