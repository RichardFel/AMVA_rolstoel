# %%
# TODO
# sample weights

# Modules
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import logging
from scipy import stats

logging.getLogger("tensorflow").setLevel(
    logging.ERROR)  # this goes *before* tf import


def predict_activities_2groups_ankle():
    charact = pd.read_excel(
        'Characteristics/Participantkarakteristieken_Rolstoel_v1.xlsx', sheet_name='Opgeschoond')
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

    # Settings
    # charact = pd.read_excel(
    #     'Characteristics/AMVA_karakteristieken_lopend_def.xlsx')
    measurements = {}
    path = 'processed_data_2groups'
    Dimensions = 6
    Activity = 50

    # %%
    unique_pt = np.array([])
    for file in os.listdir(path):
        if file.endswith('.csv'):
            unique_pt = np.append(unique_pt, file.split('_')[0])
    unique_pt = np.unique(unique_pt)
    # Load data into one location
    total_parts = 0
    for pt in unique_pt:
        data_ankle = pd.read_csv(f'{path}/{pt}_2_1.csv', index_col=0)
        # data_arm = pd.read_csv(f'{path}/{pt}_3_1.csv', index_col=0)
        data_ankle = data_ankle.reset_index(drop=True)
        # data_arm = data_arm.reset_index(drop=True)
        data = data_ankle.drop(columns=['index'])
        total_parts += len(data_ankle) // 50
        measurements[pt] = data

    data = np.empty((total_parts, Activity, Dimensions))
    outcome = np.empty((total_parts))
    participant = np.empty((total_parts))

    # Load data into numpy array
    counter = 0
    for pt in measurements:
        file_size = len(measurements[pt])
        for i in range(file_size//50):
            part = measurements[pt].loc[i*50:i*50+49]
            data[counter] = part.loc[:, 'Ax':'Gz'].values
            outcome[counter] = part.iloc[0]['Categories']
            participant[counter] = pt
            counter += 1

    # %%
    unique, counts = np.unique(outcome, return_counts=True)
    for c, freq in zip(unique, counts):
        print(f'Class: {c}, count {freq}')
    # Clean data
    sit_clean, sit_part_clean = remove_outliers(data[np.where(outcome == 0)[0], :, :],
                                                participant[np.where(outcome == 0)[0]])
    wal_clean, wal_part_clean = remove_outliers(data[np.where(outcome == 1)[0], :, :],
                                                participant[np.where(outcome == 1)[0]])
    # oth_clean, oth_part_clean = remove_outliers(data[np.where(outcome == 2)[0], :, :],
    #                                             participant[np.where(outcome == 2)[0]])

    data_clean = np.concatenate((sit_clean, wal_clean))
    part_clean = np.concatenate(
        (sit_part_clean, wal_part_clean))
    outcomes_clean = np.concatenate((np.zeros(len(sit_clean)),
                                    np.ones(len(wal_clean))))
    print(f'data_clean: {len(data_clean)}')
    print(f'part_clean: {len(part_clean)}')
    print(f'outcomes_clean: {len(outcomes_clean)}')
    num_categories = len(np.unique(outcomes_clean, return_counts=True)[0])
    y_one_hot = to_categorical(outcomes_clean, num_classes=num_categories)

    # %%
    # Definition of the model
    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=15, restore_best_weights=False)

    # Build the neural network

    def define_model():
        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=(
                Activity, Dimensions)),
            MaxPooling1D(2),
            BatchNormalization(),

            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            BatchNormalization(),

            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            BatchNormalization(),

            Flatten(),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_categories, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    # %%
    # Settings
    results_all = {}
    results_walking = {}
    results_Nwalking = {}

    unique_part = np.unique(part_clean)
    walking = charact.loc[(charact['Onderzoeksgroep'] == 3),
                          'Proefpersoonnummer'].unique()
    walking = unique_part[np.isin(unique_part, walking)]
    nWalking = charact.loc[(charact['Onderzoeksgroep'] == 4),
                           'Proefpersoonnummer'].unique()
    nWalking = unique_part[np.isin(unique_part, nWalking)]

    unique_part = np.unique(part_clean)
    batch_size = 3
    repeat = 10
    groups = {'unique_part': unique_part,
              'walking': walking,
              'nWalking': nWalking, }
    # Split 20/80 for the train/validate set
    data_split = int((len(unique_part) - batch_size) / 5)

    # Repeated all
    for i in range(repeat):
        for name, group in groups.items():

            # Settings
            selected_participants = []
            participants_remaining = np.copy(group)
            counter = 0

            # select 5 participants
            while len(participants_remaining) > (batch_size - 1):
                counter += 1
                print(
                    f'Round ({i}/{repeat - 1}), batch ({counter}/{len(group) // batch_size })')

                np.random.shuffle(participants_remaining)
                selected_batch = participants_remaining[:batch_size]
                participants_remaining = participants_remaining[batch_size:]

                # Indices train, validate, test
                train_test_part = np.unique(
                    part_clean[~np.isin(part_clean, selected_batch)])
                selection = np.random.choice(
                    train_test_part, data_split, replace=False)
                idx_train_data = np.where(
                    (~np.isin(part_clean, selection)) &
                    (~np.isin(part_clean, selected_batch)))[0]
                idx_validate_data = np.where(
                    (np.isin(part_clean, selection)))[0]
                idx_test_data = np.where(
                    np.isin(part_clean, selected_batch))[0]

                train_data = data_clean[idx_train_data, :, :]
                train_outcome = y_one_hot[idx_train_data]
                train_data, train_outcome = shuffle(train_data, train_outcome)

                validate_data = data_clean[idx_validate_data, :, :]
                validate_outcome = y_one_hot[idx_validate_data]
                validate_data, validate_outcome = shuffle(
                    validate_data, validate_outcome)

                test_data = data_clean[idx_test_data, :, :]
                test_outcome = y_one_hot[idx_test_data]
                test_data, test_outcome = shuffle(test_data, test_outcome)

                # Calculate class weights
                class_weights = compute_class_weight(class_weight="balanced",
                                                     classes=np.unique(
                                                         outcomes_clean[idx_train_data]),
                                                     y=outcomes_clean[idx_train_data]
                                                     )
                class_weight_dict = dict(enumerate(class_weights))

                # Compile the model
                model = define_model()
                tf.autograph.set_verbosity(0)
                # Train the model
                history = model.fit(train_data, train_outcome,
                                    epochs=150, batch_size=32,
                                    validation_data=(
                                        validate_data, validate_outcome),
                                    callbacks=[early_stopping], shuffle=True,
                                    # class_weight=class_weight_dict,
                                    verbose=False)

                test_predictions = model.predict(test_data)
                # Convert one-hot encoded predictions to labels
                test_predictions = np.argmax(test_predictions, axis=1)
                test_labels = np.argmax(test_outcome, axis=1)

                overall_accuracy = accuracy_score(
                    test_labels, test_predictions)
                weighted_recall = recall_score(
                    test_labels, test_predictions, average='macro')
                weighted_precision = precision_score(
                    test_labels, test_predictions, average='macro')
                weighted_f1_score = f1_score(
                    test_labels, test_predictions, average='macro')
                confusion_mat = confusion_matrix(test_labels, test_predictions)
                normalized_confusion_mat = np.round(confusion_mat.astype(
                    'float') / confusion_mat.sum(axis=1)[:, np.newaxis], 2)

                train_acc = history.history['accuracy']
                val_acc = history.history['val_accuracy']
                n_epochs = len(train_acc)

                if name == 'walking':
                    results_walking[f'{i}_{counter}'] = [
                        selected_batch, selection, n_epochs, train_acc, val_acc, overall_accuracy,
                        weighted_recall, weighted_precision, weighted_f1_score, confusion_mat, normalized_confusion_mat]
                elif name == 'nWalking':
                    results_Nwalking[f'{i}_{counter}'] = [
                        selected_batch, selection, n_epochs, train_acc, val_acc, overall_accuracy,
                        weighted_recall, weighted_precision, weighted_f1_score, confusion_mat, normalized_confusion_mat]
                else:
                    results_all[f'{i}_{counter}'] = [
                        selected_batch, selection, n_epochs, train_acc, val_acc, overall_accuracy,
                        weighted_recall, weighted_precision, weighted_f1_score, confusion_mat, normalized_confusion_mat]

                # for part in selected_batch:
                #     # Select individual
                #     idx_tmp_data = np.where(np.isin(part_clean, part))[0]
                #     tmp_data = data_clean[idx_tmp_data, :, :]
                #     tmp_outcome = y_one_hot[idx_tmp_data]
                #     tmp_data, tmp_outcome = shuffle(tmp_data, tmp_outcome)
                #     tmp_predictions = model.predict(tmp_data)

                #     # Convert one-hot encoded predictions to labels
                #     tmp_predictions = np.argmax(tmp_predictions, axis=1)
                #     tmp_labels = np.argmax(tmp_outcome, axis=1)

                #     # Calculate outcomes per individual
                #     overall_accuracy = accuracy_score(tmp_labels, tmp_predictions)
                #     part_data = charact.loc[charact['Proefpersoonnummer'] == part, [
                #         'Onderzoeksgroep', 'Leeftijd']].values[0]

    results_all = pd.DataFrame.from_dict(results_all, orient='index', columns=['selected_batch', 'selection', 'n_epochs', 'train_acc',
                                                                               'val_acc', 'overall_accuracy', 'weighted_recall',
                                                                               'weighted_precision', 'weighted_f1_score', 'confusion_mat',
                                                                               'normalized_confusion_mat'])
    results_all.to_excel('Results/Results_ankle_l2o.xlsx')

    results_walking = pd.DataFrame.from_dict(results_walking, orient='index', columns=['selected_batch', 'selection', 'n_epochs', 'train_acc',
                                                                                       'val_acc', 'overall_accuracy', 'weighted_recall',
                                                                                       'weighted_precision', 'weighted_f1_score', 'confusion_mat',
                                                                                       'normalized_confusion_mat'])
    results_walking.to_excel('Results/results_walking_ankle_l2o.xlsx')

    results_Nwalking = pd.DataFrame.from_dict(results_Nwalking, orient='index', columns=['selected_batch', 'selection', 'n_epochs', 'train_acc',
                                                                                         'val_acc', 'overall_accuracy', 'weighted_recall',
                                                                                         'weighted_precision', 'weighted_f1_score', 'confusion_mat',
                                                                                         'normalized_confusion_mat'])
    results_Nwalking.to_excel('Results/results_Nwalking_ankle_l2o.xlsx')

    # %%
    # Results all data

    results = {'results_all': results_all,
               'results_walking': results_walking,
               'results_Nwalking': results_Nwalking,
               }

    confidence = 0.95

    for name, df in results.items():
        print(name)
        avg_acc = round(df['overall_accuracy'].mean() * 100, 1)
        std_acc = round(df['overall_accuracy'].std() * 100, 1)
        # sem = round(stats.sem(df['overall_accuracy']) * 100, 1)
        # ci_acc = np.round(stats.t.interval(confidence, len(
        #     df['overall_accuracy'])-1, loc=avg_acc, scale=sem), 1)
        min_acc = round(df['overall_accuracy'].min() * 100, 1)
        max_acc = round(df['overall_accuracy'].max() * 100, 1)

        avg_f1 = round(df['weighted_f1_score'].mean() * 100, 1)
        std_f1 = round(df['weighted_f1_score'].std() * 100, 1)
        # sem = round(stats.sem(df['weighted_f1_score']) * 100, 1)
        # ci_f1 = np.round(stats.t.interval(confidence, len(
        #     df['weighted_f1_score'])-1, loc=avg_acc, scale=sem) * 100, 1)

        min_f1 = round(df['weighted_f1_score'].min() * 100, 1)
        max_f1 = round(df['weighted_f1_score'].max() * 100, 1)

        avg_recall = round(df['weighted_recall'].mean() * 100, 1)
        std_recall = round(df['weighted_recall'].std() * 100, 1)
        # sem = round(stats.sem(df['weighted_recall']) * 100, 1)
        # ci_recall = np.round(stats.t.interval(confidence, len(
        #     df['weighted_recall'])-1, loc=avg_acc, scale=sem) * 100, 1)

        min_recall = round(df['weighted_recall'].min() * 100, 1)
        max_recall = round(df['weighted_recall'].max() * 100, 1)

        avg_precision = round(df['weighted_precision'].mean() * 100, 1)
        std_precision = round(df['weighted_precision'].std() * 100, 1)
        # sem = round(stats.sem(df['weighted_precision']) * 100, 1)
        # ci_precision = np.round(stats.t.interval(confidence, len(
        #     df['weighted_precision'])-1, loc=avg_acc, scale=sem) * 100, 1)

        min_precision = round(df['weighted_precision'].min() * 100, 1)
        max_precision = round(df['weighted_precision'].max() * 100, 1)

        print(f'Accuracy {avg_acc} ({std_acc}), [{min_acc},{max_acc}]')
        print(f'F1 {avg_f1} ({std_f1}), [{min_f1},{max_f1}]')
        print(
            f'Recall {avg_recall} ({std_recall}), [{min_recall},{max_recall}]')
        print(
            f'Precision {avg_precision} ({std_precision}), [{min_precision},{max_precision}]')
        normalized_confusion_mat = df['normalized_confusion_mat']
        for idx, matrix in normalized_confusion_mat.items():
            for j in matrix:
                if np.isnan(j).any():
                    normalized_confusion_mat.drop(index=idx, inplace=True)

        normalized_confusion_mat = (
            np.mean(normalized_confusion_mat)*100).astype(int)
        # Create a heatmap for the confusion matrix
        sns.set(font_scale=1.4)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(normalized_confusion_mat, annot=True,  cmap="Blues", cbar=False, square=True,
                    xticklabels=["Stationary",
                                 "Locomotion"],
                    yticklabels=["Stationary", "Locomotion"], ax=ax)
        ax.set_xlabel("Predicted Categories")
        ax.set_ylabel("Observed Categories")

        if name == 'results_all':
            ax.set_title(f"A. All children")
        elif name == 'results_walking':
            ax.set_title(f"C. Walking & Wheeling")
        else:
            ax.set_title(f"B. Wheeling")

        fig.tight_layout()
        fig.savefig(
            f'Figures/Confusion_matrix_2groups_ankle_{name}.png', dpi=400)

# %%
