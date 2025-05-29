import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from imblearn.over_sampling import SMOTE

# --- 0. parameters ---
EXPERIMENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"dnn_experiment_{EXPERIMENT_TIMESTAMP}"

TARGET_COLUMN = 'law_level'

EXPECTED_FEATURE_COLUMNS = [
    'Latitude', 'Longitude', 'weekday_sin', 'weekday_cos', 'hour_sin', 'hour_cos',
    'apparent_temperature', 'precipitation', 'median_income',
    'unemployment_rate', 'no_high_school_ratio', 'bachelor_or_higher_ratio',
    'poverty_rate', 'population_density', 'poi_count_nightlife', 'poi_count_food_drink',
    'poi_count_retail_convenience', 'poi_count_retail_risky', 'poi_count_transport_hub',
    'poi_count_education', 'poi_count_finance', 'poi_count_public_safety', 'poi_count_healthcare',
    'poi_count_recreation_open', 'poi_count_recreation_indoor'
]

HYPERPARAMETERS = {
    'apply_smote': True,
    'smote_k_neighbors_max': 5,
    'smote_sampling_strategy': 'auto',
    'test_size': 0.2,
    'val_size': 0.25,
    'random_state': 42,
    'batch_size': 32,
    'epochs': 300,
    'learning_rate': 1e-4,
    'patience_epochs': 15,
    'gradient_clip_max_norm': 1.0,
    'model_architecture': {
        'layer_1_neurons': 64,
        'dropout_1': 0.3,
        'layer_2_neurons': 128,
        'dropout_2': 0.3,
        'layer_3_neurons': 128,
        'dropout_3': 0.3,
        'layer_4_neurons': 256,
        'dropout_4': 0.3,
        'layer_5_neurons': 512,
        'dropout_5': 0.2,
        'layer_6_neurons': 1024,
        'dropout_6': 0.2,
        'layer_7_neurons': 512,
        'dropout_7': 0.2,
        'layer_8_neurons': 256,
        'dropout_8': 0.2,
        'layer_9_neurons': 128,
        'dropout_9': 0.2,
        'layer_10_neurons': 64,
        'dropout_10': 0.2,
        'layer_11_neurons': 64,
        'dropout_11': 0.2,
        'layer_12_neurons': 32,
        'dropout_12': 0.2,
        'layer_13_neurons': 32,
        'dropout_13': 0.2,
        'layer_14_neurons': 16,
        'dropout_14': 0.2,
        'layer_15_neurons': 16,
        'dropout_15': 0.1,

        'activation': 'ReLU'
    },
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss'
}

df_full = pd.read_csv("DNN_DATA_with_LAW_LABEL_and_OSM_POI.csv")
print(f"---0. Label Distribution in the Full Dataset (df_full) ---")
if TARGET_COLUMN in df_full.columns:
    label_counts_full = df_full[TARGET_COLUMN].value_counts().sort_index()
    label_proportions_full = df_full[TARGET_COLUMN].value_counts(normalize=True).sort_index()

    print("\nCounts for each label in the full dataset:")
    print(label_counts_full)
    print("\nProportions for each label in the full dataset:")
    print(label_proportions_full)

print(f"Initial data shape: {df_full.shape}")

# --- 1. load and process data ---
try:
    print("--- 1. Loading Data ---")
    print(f"Initial data shape: {df_full.shape}")

    # remove 'Unnamed: 0' column (if exist)
    if 'Unnamed: 0' in df_full.columns:
        print("Dropping 'Unnamed: 0' column.")
        df_full = df_full.drop(columns=['Unnamed: 0'])

    # make sure all data existed
    all_required_cols_for_start = EXPECTED_FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols_at_start = [col for col in all_required_cols_for_start if col not in df_full.columns]
    if missing_cols_at_start:
        raise ValueError(f"CRITICAL: Missing expected columns in DNN_DATA.csv: {missing_cols_at_start}")


    current_cols = df_full.columns.tolist()
    cols_to_keep = [col for col in current_cols if col in all_required_cols_for_start]
    if len(cols_to_keep) < len(current_cols):
        dropped_extra_cols = [col for col in current_cols if col not in cols_to_keep]
        print(f"Warning: Dropping extra columns not in EXPECTED_FEATURE_COLUMNS or TARGET_COLUMN: {dropped_extra_cols}")
        df_full = df_full[cols_to_keep]

    print(f"Data shape after initial column filtering: {df_full.shape}")
    print("Columns used for modeling (initially):", df_full.columns.tolist())

    # ---  NaN process( df_full ) ---

    print("\n--- Applying Initial NaN Handling on df_full (before splitting) ---")
    cols_to_dropna_subset = ['tensor_row', 'tensor_col']

    cols_for_dropna_from_user = [
        'Latitude', 'Longitude', 'median_income', 'unemployment_rate', 'no_high_school_ratio',
        'bachelor_or_higher_ratio', 'poverty_rate', 'population_density',
        'apparent_temperature', 'precipitation','danger_level'
    ]

    actual_cols_for_dropna = [col for col in cols_for_dropna_from_user if col in df_full.columns]

    initial_rows = len(df_full)
    df_full.dropna(subset=actual_cols_for_dropna, inplace=True)
    if len(df_full) < initial_rows:
        print(f"Dropped {initial_rows - len(df_full)} rows due to NaNs in specified columns: {actual_cols_for_dropna}.")

    print(f"Shape after initial NaN drop: {df_full.shape}")
    if df_full.empty:
        raise ValueError("DataFrame is empty after initial NaN drop. Check your data or dropna subset.")

except FileNotFoundError:
    print("Error: DNN_DATA.csv not found. Please ensure the file is uploaded correctly.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading or initial cleaning: {e}")
    exit()


print("\n--- Final NaN/Inf Check on df_full before preprocessing function call ---")
if df_full[EXPECTED_FEATURE_COLUMNS].isnull().any().any():
    print("CRITICAL WARNING: NaNs found in feature columns of df_full!")
    print(df_full[EXPECTED_FEATURE_COLUMNS].isnull().sum()[df_full[EXPECTED_FEATURE_COLUMNS].isnull().sum() > 0])
    print("Exiting. Please ensure df_full has no NaNs in features before calling preprocess_data.")
    exit()
if np.isinf(df_full[EXPECTED_FEATURE_COLUMNS].select_dtypes(include=np.number).values).any():
    print("CRITICAL WARNING: Infs found in feature columns of df_full!")

    print("Exiting. Please ensure df_full has no Infs in features.")
    exit()
if df_full['law_level'].isnull().any().any():
    print("there are data without label")
else:
    print("df_full features are clean of NaNs and Infs before calling preprocess_data.")


# --- 2. data preprocess ---
def preprocess_data(df, target_column, feature_columns_list_input,  # 重命名以區分
                    test_size_prop, val_size_prop, random_state_seed, apply_smote=False,smote_random_state=None):
    print("\n--- 2. Starting Data Preprocessing ---")


    feature_columns_list = list(feature_columns_list_input)

    X = df[feature_columns_list]
    y_original = df[target_column]


    unique_y_values = np.sort(y_original.unique())
    is_correctly_encoded = (len(unique_y_values) > 0 and  # 確保y不是空的
                            unique_y_values[0] == 0 and
                            np.all(np.diff(unique_y_values) == 1) and
                            unique_y_values[-1] == len(unique_y_values) - 1)

    if not is_correctly_encoded:
        print(
            f"Warning: Target variable '{target_column}' unique values are {unique_y_values}. Mapping to 0-based index.")
        label_mapping = {label: i for i, label in enumerate(unique_y_values)}
        y = y_original.map(label_mapping)
        print(f"Mapped unique target values: {np.sort(y.unique())}")
        if y.isnull().any():
            raise ValueError("NaNs introduced in target variable after mapping. Original values: {y_original.unique()}")
        HYPERPARAMETERS['label_mapping'] = label_mapping  # 記錄映射關係
    else:
        y = y_original
        print(f"Target variable '{target_column}' unique values {unique_y_values} are correctly 0-indexed.")
        HYPERPARAMETERS['label_mapping'] = "Original labels were already 0-indexed and sequential."

    # data split (testing data)
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size_prop, random_state=random_state_seed, stratify=y
        )
    except ValueError as e:
        print(f"ValueError during first split (stratify might be an issue if y_temp is small or has few classes): {e}")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size_prop, random_state=random_state_seed
        )
    print(f"Shape after first split: X_temp={X_temp.shape}, X_test={X_test.shape}")

    # split 2 (training and validation set)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_prop, random_state=random_state_seed, stratify=y_temp
        )
    except ValueError as e:
        print(f"ValueError during second split (stratify might be an issue if y_temp is small or has few classes): {e}")
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_prop, random_state=random_state_seed
        )
    print(f"Shape after second split: X_train={X_train.shape}, X_val={X_val.shape}")

    if apply_smote:
        print("\n--- Applying SMOTE to the training set ---")

        min_samples_in_class = y_train.value_counts().min()
        k_neighbors_smote = min(5, min_samples_in_class - 1) if min_samples_in_class > 1 else 1

        if k_neighbors_smote < 1:
            print(
                f"Warning: Cannot apply SMOTE as k_neighbors would be < 1 (min_samples_in_class={min_samples_in_class}). Skipping SMOTE.")
        else:
            try:

                smote = SMOTE(random_state=smote_random_state if smote_random_state is not None else random_state_seed,
                              sampling_strategy='auto',  # 或者 'not majority'
                              k_neighbors=k_neighbors_smote)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"Shapes after SMOTE: X_train={X_train.shape}, y_train={y_train.shape}")
                print(f"y_train value counts after SMOTE:\n{y_train.value_counts().sort_index()}")
                HYPERPARAMETERS['smote_applied'] = True
                HYPERPARAMETERS['smote_k_neighbors'] = k_neighbors_smote
                HYPERPARAMETERS['smote_sampling_strategy'] = smote.sampling_strategy_  # 記錄實際使用的策略
            except Exception as e_smote:
                print(f"Error applying SMOTE: {e_smote}. Proceeding without SMOTE.")
                HYPERPARAMETERS['smote_applied'] = False
    else:
        HYPERPARAMETERS['smote_applied'] = False

    print("\n--- 2.4 Outlier Handling (based on X_train stats) ---")
    X_train_no_outliers = X_train.copy()
    X_val_no_outliers = X_val.copy()
    X_test_no_outliers = X_test.copy()

    outlier_bounds = {}
    for column in feature_columns_list:
        Q1 = X_train[column].quantile(0.25)
        Q3 = X_train[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_bounds[column] = {'lower': lower_bound, 'upper': upper_bound}

        X_train_no_outliers[column] = np.where(X_train_no_outliers[column] < lower_bound, lower_bound,
                                               X_train_no_outliers[column])
        X_train_no_outliers[column] = np.where(X_train_no_outliers[column] > upper_bound, upper_bound,
                                               X_train_no_outliers[column])

        X_val_no_outliers[column] = np.where(X_val_no_outliers[column] < lower_bound, lower_bound,
                                             X_val_no_outliers[column])
        X_val_no_outliers[column] = np.where(X_val_no_outliers[column] > upper_bound, upper_bound,
                                             X_val_no_outliers[column])

        X_test_no_outliers[column] = np.where(X_test_no_outliers[column] < lower_bound, lower_bound,
                                              X_test_no_outliers[column])
        X_test_no_outliers[column] = np.where(X_test_no_outliers[column] > upper_bound, upper_bound,
                                              X_test_no_outliers[column])
    print("Outlier handling complete.")

    print("\n--- 2.5 Feature Scaling (StandardScaler) ---")
    scaler = StandardScaler()


    stds = X_train_no_outliers[feature_columns_list].std()
    cols_zero_std = stds[stds < 1e-9].index.tolist()

    final_features_for_scaling = list(feature_columns_list)

    if cols_zero_std:
        print(
            f"CRITICAL WARNING: Columns with zero/very small standard deviation in training set (after outlier handling): {cols_zero_std}")
        print("These columns will be dropped before scaling.")
        X_train_no_outliers = X_train_no_outliers.drop(columns=cols_zero_std)
        X_val_no_outliers = X_val_no_outliers.drop(columns=cols_zero_std)
        X_test_no_outliers = X_test_no_outliers.drop(columns=cols_zero_std)
        final_features_for_scaling = [col for col in feature_columns_list if col not in cols_zero_std]  # update
        if not final_features_for_scaling:
            raise ValueError("All feature columns were dropped due to zero standard deviation. Cannot proceed.")
        print(f"Remaining features after dropping zero_std_cols: {final_features_for_scaling}")


    scaler.fit(X_train_no_outliers[final_features_for_scaling])

    X_train_scaled_array = scaler.transform(X_train_no_outliers[final_features_for_scaling])
    X_train_scaled_df = pd.DataFrame(X_train_scaled_array, columns=final_features_for_scaling, index=X_train.index)

    X_val_scaled_array = scaler.transform(X_val_no_outliers[final_features_for_scaling])
    X_val_scaled_df = pd.DataFrame(X_val_scaled_array, columns=final_features_for_scaling, index=X_val.index)

    X_test_scaled_array = scaler.transform(X_test_no_outliers[final_features_for_scaling])
    X_test_scaled_df = pd.DataFrame(X_test_scaled_array, columns=final_features_for_scaling, index=X_test.index)
    print("Feature scaling complete.")

    for name, df_check in {"X_train_scaled_df": X_train_scaled_df, "X_val_scaled_df": X_val_scaled_df,
                           "X_test_scaled_df": X_test_scaled_df}.items():
        if df_check.isnull().any().any() or np.isinf(df_check.values).any():
            print(
                f"CRITICAL ERROR: NaN or Inf found in {name} AFTER scaling! Check source data and preprocessing logic.")
            print(f"{name} NaNs:\n{df_check.isnull().sum()[df_check.isnull().sum() > 0]}")
            inf_sum_col = np.isinf(df_check.values).sum(axis=0)
            if np.any(inf_sum_col > 0):
                print(
                    f"{name} Infs per column:\n{pd.Series(inf_sum_col, index=df_check.columns)[pd.Series(inf_sum_col, index=df_check.columns) > 0]}")


    return X_train_scaled_df, X_val_scaled_df, X_test_scaled_df, \
        y_train, y_val, y_test, \
        scaler, outlier_bounds, final_features_for_scaling



try:
    X_train_p, X_val_p, X_test_p, \
        y_train_p, y_val_p, y_test_p, \
        fitted_scaler, fitted_outlier_bounds, final_feature_columns = \
        preprocess_data(
            df_full,
            target_column=TARGET_COLUMN,
            feature_columns_list_input=list(EXPECTED_FEATURE_COLUMNS),  # 傳入副本
            test_size_prop=HYPERPARAMETERS['test_size'],
            val_size_prop=HYPERPARAMETERS['val_size'],
            random_state_seed=HYPERPARAMETERS['random_state'],
            apply_smote=HYPERPARAMETERS.get('apply_smote', False),
            # smote_random_state 可以沿用 random_state_seed
            smote_random_state=HYPERPARAMETERS.get('smote_random_state', HYPERPARAMETERS['random_state'])
        )
    print("\nPreprocessing function executed.")
    print(f"Final features used for model: {final_feature_columns}")
    HYPERPARAMETERS['final_input_features'] = final_feature_columns
    HYPERPARAMETERS['num_final_features'] = len(final_feature_columns)
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit()

# --- 3. transfer to PyTorch Tensors and create DataLoader ---
print("\n--- 3. Converting to PyTorch Tensors and Creating DataLoaders ---")
try:
    X_train_tensor = torch.FloatTensor(X_train_p[final_feature_columns].values)
    X_val_tensor = torch.FloatTensor(X_val_p[final_feature_columns].values)
    X_test_tensor = torch.FloatTensor(X_test_p[final_feature_columns].values)

    y_train_tensor = torch.LongTensor(y_train_p.values)
    y_val_tensor = torch.LongTensor(y_val_p.values)
    y_test_tensor = torch.LongTensor(y_test_p.values)

    for name, tensor_check in {"X_train_tensor": X_train_tensor, "X_val_tensor": X_val_tensor,
                               "X_test_tensor": X_test_tensor}.items():
        if torch.isnan(tensor_check).any() or torch.isinf(tensor_check).any():
            print(f"CRITICAL ERROR: NaN or Inf found in {name} PyTorch Tensor! Exiting.")
            exit()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(dataset=val_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=HYPERPARAMETERS['batch_size'], shuffle=False)
    print("DataLoaders created.")
except Exception as e:
    print(f"Error during Tensor conversion or DataLoader creation: {e}")
    exit()

# --- 4. define DNN model ---
print("\n--- 4. Defining DNN Model ---")

input_dim = len(final_feature_columns)

if len(y_train_tensor) == 0:
    raise ValueError("y_train_tensor is empty, cannot determine num_classes. Check data splitting and NaN handling.")
num_classes = len(torch.unique(y_train_tensor))

HYPERPARAMETERS['input_dim_actual'] = input_dim
HYPERPARAMETERS['num_classes_actual'] = num_classes

print(f"Model input dimension (actual): {input_dim}")
print(f"Number of classes (actual): {num_classes}")
if num_classes <= 1:
    print("Error: Number of classes is <= 1. Check target variable encoding and data after splits.")
    exit()


class DNNModel(nn.Module):
    def __init__(self, model_input_dim, model_num_classes, model_arch_params):  # 更改參數名
        super(DNNModel, self).__init__()

        self.layer_1 = nn.Linear(model_input_dim, model_arch_params['layer_1_neurons'])
        self.bn1 = nn.BatchNorm1d(model_arch_params['layer_1_neurons'])
        self.dropout1 = nn.Dropout(model_arch_params['dropout_1'])

        self.layer_2 = nn.Linear(model_arch_params['layer_1_neurons'], model_arch_params['layer_2_neurons'])
        self.bn2 = nn.BatchNorm1d(model_arch_params['layer_2_neurons'])
        self.dropout2 = nn.Dropout(model_arch_params['dropout_2'])

        self.layer_3 = nn.Linear(model_arch_params['layer_2_neurons'], model_arch_params['layer_3_neurons'])
        self.bn3 = nn.BatchNorm1d(model_arch_params['layer_3_neurons'])
        self.dropout3 = nn.Dropout(model_arch_params['dropout_3'])

        self.layer_4 = nn.Linear(model_arch_params['layer_3_neurons'], model_arch_params['layer_4_neurons'])
        self.bn4 = nn.BatchNorm1d(model_arch_params['layer_4_neurons'])
        self.dropout4 = nn.Dropout(model_arch_params['dropout_4'])

        self.layer_5 = nn.Linear(model_arch_params['layer_4_neurons'], model_arch_params['layer_5_neurons'])
        self.bn5 = nn.BatchNorm1d(model_arch_params['layer_5_neurons'])
        self.dropout5 = nn.Dropout(model_arch_params['dropout_5'])

        self.layer_6 = nn.Linear(model_arch_params['layer_5_neurons'], model_arch_params['layer_6_neurons'])
        self.bn6 = nn.BatchNorm1d(model_arch_params['layer_6_neurons'])
        self.dropout6 = nn.Dropout(model_arch_params['dropout_6'])

        self.layer_7 = nn.Linear(model_arch_params['layer_6_neurons'], model_arch_params['layer_7_neurons'])
        self.bn7 = nn.BatchNorm1d(model_arch_params['layer_7_neurons'])
        self.dropout7 = nn.Dropout(model_arch_params['dropout_7'])

        self.layer_8 = nn.Linear(model_arch_params['layer_7_neurons'], model_arch_params['layer_8_neurons'])
        self.bn8 = nn.BatchNorm1d(model_arch_params['layer_8_neurons'])
        self.dropout8 = nn.Dropout(model_arch_params['dropout_8'])

        self.layer_9 = nn.Linear(model_arch_params['layer_8_neurons'], model_arch_params['layer_9_neurons'])
        self.bn9 = nn.BatchNorm1d(model_arch_params['layer_9_neurons'])
        self.dropout9 = nn.Dropout(model_arch_params['dropout_9'])

        self.layer_10 = nn.Linear(model_arch_params['layer_9_neurons'], model_arch_params['layer_10_neurons'])
        self.bn10 = nn.BatchNorm1d(model_arch_params['layer_10_neurons'])
        self.dropout10 = nn.Dropout(model_arch_params['dropout_10'])

        self.layer_11 = nn.Linear(model_arch_params['layer_10_neurons'], model_arch_params['layer_11_neurons'])
        self.bn11 = nn.BatchNorm1d(model_arch_params['layer_11_neurons'])
        self.dropout11 = nn.Dropout(model_arch_params['dropout_11'])

        self.layer_12 = nn.Linear(model_arch_params['layer_11_neurons'], model_arch_params['layer_12_neurons'])
        self.bn12 = nn.BatchNorm1d(model_arch_params['layer_12_neurons'])
        self.dropout12 = nn.Dropout(model_arch_params['dropout_12'])

        self.layer_13 = nn.Linear(model_arch_params['layer_12_neurons'], model_arch_params['layer_13_neurons'])
        self.bn13 = nn.BatchNorm1d(model_arch_params['layer_13_neurons'])
        self.dropout13 = nn.Dropout(model_arch_params['dropout_13'])

        self.layer_14 = nn.Linear(model_arch_params['layer_13_neurons'], model_arch_params['layer_14_neurons'])
        self.bn14 = nn.BatchNorm1d(model_arch_params['layer_14_neurons'])
        self.dropout14 = nn.Dropout(model_arch_params['dropout_14'])

        self.layer_15 = nn.Linear(model_arch_params['layer_14_neurons'], model_arch_params['layer_15_neurons'])
        self.bn15 = nn.BatchNorm1d(model_arch_params['layer_15_neurons'])
        self.dropout15 = nn.Dropout(model_arch_params['dropout_15'])

        self.output_layer = nn.Linear(model_arch_params['layer_15_neurons'], model_num_classes)  # 使用 model_num_classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.relu(self.layer_2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.relu(self.layer_3(x))
        x = self.bn3(x)
        x = self.dropout3(x)

        x = self.relu(self.layer_4(x))
        x = self.bn4(x)
        x = self.dropout4(x)

        x = self.relu(self.layer_5(x))
        x = self.bn5(x)
        x = self.dropout5(x)

        x = self.relu(self.layer_6(x))
        x = self.bn6(x)
        x = self.dropout6(x)

        x = self.relu(self.layer_7(x))
        x = self.bn7(x)
        x = self.dropout7(x)

        x = self.relu(self.layer_8(x))
        x = self.bn8(x)
        x = self.dropout8(x)

        x = self.relu(self.layer_9(x))
        x = self.bn9(x)
        x = self.dropout9(x)

        x = self.relu(self.layer_10(x))
        x = self.bn10(x)
        x = self.dropout10(x)

        x = self.relu(self.layer_11(x))
        x = self.bn11(x)
        x = self.dropout11(x)

        x = self.relu(self.layer_12(x))
        x = self.bn12(x)
        x = self.dropout12(x)

        x = self.relu(self.layer_13(x))
        x = self.bn13(x)
        x = self.dropout13(x)

        x = self.relu(self.layer_14(x))
        x = self.bn14(x)
        x = self.dropout14(x)

        x = self.relu(self.layer_15(x))
        x = self.bn15(x)
        x = self.dropout15(x)

        x = self.output_layer(x)
        return x


device = torch.device("cpu")

model = DNNModel(input_dim, num_classes, HYPERPARAMETERS['model_architecture']).to(device)
print(model)
print(f"Using device: {device}")

# --- 5. define loss function and optimizer ---
print("\n--- 5. Defining Loss Function and Optimizer ---")
unique_classes_for_weight = np.unique(y_train_p.values)
print(f"Unique classes in y_train_p for weight calculation: {unique_classes_for_weight}")

expected = np.arange(0, num_classes)  # [1, 2, 3]
if not np.array_equal(unique_classes_for_weight, expected):
    print(
        f"Warning: Classes should be {expected.tolist()}, but got {unique_classes_for_weight.tolist()}."
    )


class_weights_array = compute_class_weight(
    'balanced',
    classes=unique_classes_for_weight,
    y=y_train_p.values
)
class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float).to(device)
print(f"Calculated class weights: {class_weights_tensor.cpu().tolist()}")
HYPERPARAMETERS['class_weights_used'] = class_weights_tensor.cpu().tolist()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMETERS['learning_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.2, min_lr=1e-6)
print(
    f"Optimizer: {HYPERPARAMETERS['optimizer']}, Initial LR: {HYPERPARAMETERS['learning_rate']}, Loss: CrossEntropyLoss")

# --- 6. training and validation loop ---
print("\n--- 6. Starting Training and Validation ---")
best_val_loss = float('inf')
patience_counter = 0
actual_epochs_trained = 0

train_losses_history = []
val_losses_history = []
train_accuracies_history = []
val_accuracies_history = []

SAVED_MODEL_DIR = f"saved_models/{EXPERIMENT_NAME}"
if not os.path.exists(SAVED_MODEL_DIR):
    os.makedirs(SAVED_MODEL_DIR)
BEST_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, 'best_dnn_model_pytorch.pth')

for epoch in range(HYPERPARAMETERS['epochs']):
    actual_epochs_trained = epoch + 1
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"CRITICAL: NaN/Inf detected in model outputs at Epoch {epoch + 1}, Batch {i + 1}. Stopping.")
            exit()
        loss = criterion(outputs, labels)



        if torch.isnan(loss) or torch.isinf(loss):
            print(f"CRITICAL: NaN/Inf loss detected at Epoch {epoch + 1}, Batch {i + 1}. Stopping.")
            exit()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=HYPERPARAMETERS['gradient_clip_max_norm'])
        optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    epoch_train_loss = running_train_loss / total_train if total_train > 0 else float('nan')  # 避免除以0
    epoch_train_accuracy = correct_train / total_train if total_train > 0 else 0.0
    train_losses_history.append(epoch_train_loss)
    train_accuracies_history.append(epoch_train_accuracy)

    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)
            _, predicted_val = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()

    epoch_val_loss = running_val_loss / total_val if total_val > 0 else float('inf')  # 避免除以0, 用inf觸發保存
    epoch_val_accuracy = correct_val / total_val if total_val > 0 else 0.0
    val_losses_history.append(epoch_val_loss)
    val_accuracies_history.append(epoch_val_accuracy)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch + 1}/{HYPERPARAMETERS['epochs']}], "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}, "
          f"LR: {current_lr:.1e}")

    scheduler.step(epoch_val_loss)  # ReduceLROnPlateau

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Validation loss improved. Saved best model to {BEST_MODEL_PATH}")
    else:
        patience_counter += 1
        if patience_counter >= HYPERPARAMETERS['patience_epochs']:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break


print("\n--- 7. Evaluating on Test Set with Best Model ---")
results_summary = {}
results_summary['hyperparameters'] = HYPERPARAMETERS
results_summary['experiment_name'] = EXPERIMENT_NAME
results_summary['actual_epochs_trained'] = actual_epochs_trained
results_summary['best_validation_loss'] = best_val_loss if best_val_loss != float('inf') else None

if os.path.exists(BEST_MODEL_PATH):

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print(f"Loaded best model weights from {BEST_MODEL_PATH} to {device} for final evaluation.")
    results_summary['model_path'] = BEST_MODEL_PATH
else:
    print("Warning: No best model saved. Evaluating with the last state of the model.")
    results_summary['model_path'] = "Last state (no best model saved)"

model.eval()
all_preds_test = []
all_labels_test = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds_test.extend(predicted.cpu().numpy())
        all_labels_test.extend(labels.cpu().numpy())

if not all_labels_test:
    print("Warning: Test set is empty. Skipping test evaluation.")
    final_test_accuracy = 0.0
    report_str = "Test set was empty."
    report_dict = {}
    conf_matrix_test_list = []
else:
    final_test_accuracy = accuracy_score(all_labels_test, all_preds_test)
    print(f"\nFinal Test Accuracy: {final_test_accuracy * 100:.2f}%")

    print("\nFinal Classification Report on Test Set:")

    class_labels_for_report = [str(i) for i in range(num_classes)]


    try:
        report_str = classification_report(all_labels_test, all_preds_test, labels=range(num_classes),
                                           target_names=class_labels_for_report, zero_division=0)
        report_dict = classification_report(all_labels_test, all_preds_test, labels=range(num_classes),
                                            target_names=class_labels_for_report, zero_division=0, output_dict=True)
        print(report_str)
    except Exception as e:
        print(f"Error generating classification report (possibly due to labels): {e}")
        report_str = classification_report(all_labels_test, all_preds_test, labels=range(num_classes),
                                           zero_division=0)  # 嘗試不帶 target_names
        report_dict = classification_report(all_labels_test, all_preds_test, labels=range(num_classes), zero_division=0,
                                            output_dict=True)
        print(report_str)

    conf_matrix_test = confusion_matrix(all_labels_test, all_preds_test, labels=range(num_classes))
    conf_matrix_test_list = conf_matrix_test.tolist()
    print("\nConfusion Matrix (Test Set):")
    print(conf_matrix_test)

results_summary['test_accuracy'] = final_test_accuracy
results_summary['classification_report_test_dict'] = report_dict
results_summary['confusion_matrix_test'] = conf_matrix_test_list

# --- 8. visualization ---
print("\n--- 8. Plotting Training History ---")
HISTORY_PLOT_PATH = os.path.join(SAVED_MODEL_DIR, f'{EXPERIMENT_NAME}_history.png')


def plot_pytorch_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    if not train_losses:
        print("No training history to plot.")
        return

    actual_epochs_plot = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(actual_epochs_plot, train_losses, label='Training Loss')
    plt.plot(actual_epochs_plot, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(actual_epochs_plot, [acc * 100 for acc in train_accuracies], label='Training Accuracy (%)')
    plt.plot(actual_epochs_plot, [acc * 100 for acc in val_accuracies], label='Validation Accuracy (%)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    plt.show()


plot_pytorch_training_history(train_losses_history, val_losses_history,
                              train_accuracies_history, val_accuracies_history,
                              save_path=HISTORY_PLOT_PATH)
if HISTORY_PLOT_PATH and os.path.exists(HISTORY_PLOT_PATH):
    results_summary['training_history_plot_path'] = HISTORY_PLOT_PATH
else:
    results_summary['training_history_plot_path'] = None

# --- 9. save results ---
print("\n--- 9. Saving Hyperparameters and Results ---")
RESULTS_JSON_PATH = os.path.join(SAVED_MODEL_DIR, f'{EXPERIMENT_NAME}_summary.json')
try:
    with open(RESULTS_JSON_PATH, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"Experiment summary saved to {RESULTS_JSON_PATH}")
except TypeError as e:
    print(f"Error saving results to JSON (some data might not be serializable): {e}")



    def make_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj


    results_summary_serializable = make_serializable(results_summary)
    try:
        with open(RESULTS_JSON_PATH, 'w') as f:
            json.dump(results_summary_serializable, f, indent=4)
        print(f"Experiment summary (with serializable conversion) saved to {RESULTS_JSON_PATH}")
    except Exception as final_e:
        print(f"Could not save results summary even after serializable conversion: {final_e}")

print("\n--- Script Finished ---")