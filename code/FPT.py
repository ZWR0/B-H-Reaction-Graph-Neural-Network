import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import math
import random


def remove_highly_correlated_features(csv_file_path, corr_threshold):
    """
    Fretures Processing Tool
    
    Read CSV file, identify and remove features with correlation above a specified threshold, 
    then return the updated feature matrix DataFrame. The first column in the file serves as the index for the DataFrame.

    Parameters:
    csv_file_path (str): Path to the CSV file.
    corr_threshold (float): Threshold of correlation coefficient, used to determine which features are considered highly correlated.

    Return:
    pd.DataFrame: Updated feature matrix with highly correlated features removed.
    """

    # Read CSV file
    df = pd.read_csv(csv_file_path)
    # Remove duplicate columns, columns with identical values, and columns containing NaN values
#     df = df.loc[:, ~df.T.duplicated()]
#     df = df.loc[:, (df.nunique() != 1)]
    df = df.dropna(axis=1, how='any')

    # Compute the correlation matrix.
    correlation_matrix = df.corr(method='pearson', numeric_only=True)

    # Retaining only the upper triangular part to avoid redundancy.
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Screening for highly correlated features
    high_corr_features = upper_triangle[(upper_triangle > corr_threshold) | (upper_triangle < -corr_threshold)]

    # Initialize a set to store the names of features to be removed.
    features_to_remove = set()

    # Iterate through pairs of highly correlated features.
    for row in high_corr_features.stack().index:
        col1, _ = row
        features_to_remove.add(col1)
        
    # Create and return an updated DataFrame with highly correlated features removed.
    features_df_updated = df.drop(list(features_to_remove), axis=1)
    
    return features_df_updated

def select_top_features_with_rf_for_Regression(X, y, n_features):
    """
    Select the top n important features from the feature DataFrame based on their importance assessed by Random Forest, and return a new DataFrame.
    
    Parameters:
    X (pd.DataFrame): Feature-only DataFrame.
    y (pd.Series): Target variable Series.
    n_features (int): Number of features to retain.
    
    Return:
    pd.DataFrame: DataFrame containing only the top n most important features.
    """
    # Initialize the Random Forest Regression model
    rf_regressor = RandomForestRegressor(random_state=42)
    
    # Train the model to obtain feature importances
    rf_regressor.fit(X, y)
    
    #Obtain feature importances and sort them
    feature_importances = rf_regressor.feature_importances_
    feature_importances_sorted = sorted(zip(X.columns, feature_importances), key=lambda x: x[1], reverse=True)
    
    # Select the column names of the top n most important features
    top_n_features = [feature for feature, importance in feature_importances_sorted[:n_features]]
    
    # Select the most important features using column names
    filtered_X = X[top_n_features]
    
    return filtered_X

def calculate_cdf(data, x):
    """
    Calculate the cumulative distribution function (CDF) value of data point x within dataset data.
    
    Parameters:
    data : array-like
        A one-dimensional array or list representing a dataset.
    x : float or int
        Specific values for which the CDF (Cumulative Distribution Function) needs to be calculated.
        
    Return:
    cdf_value : float
        Cumulative probability of values less than or equal to x in the dataset.
    """
    # Calculate the mean and standard deviation of the dataset
    data_mean = np.mean(data)
    data_std = np.std(data)
    
    # Calculate the z-score of x
    z_score = (x - data_mean) / data_std
    
    # Calculate the CDF value of the normal distribution corresponding to the z-score
    cdf_value = norm.cdf(z_score)
    
    return cdf_value

def sample_distribution(df: pd.DataFrame, n: int, random_state: int = 42):
    """
    Group the input data and sample from each group according to the specified distribution type.
    
    Parameters:
    df (pd.DataFrame): Original feature data DataFrame, with the last column being the label.
    n (int): The number of groups.
    random_state (int): Random seed for reproducibility.
    
    Return:
    Tuple[pd.DataFrame, pd.DataFrame]: one with a sample count matching the Gaussian distribution of labels, and another with the unused original data.
    """
    df = df.sort_values(by=df.columns[-1], ascending=True)
    x = df.iloc[:, -1].values
    
    x_min, x_max = x.min(), x.max()
    step = (x_max - x_min) / n
    groups_boundaries = [(x_min + i*step, x_min + (i+1)*step) for i in range(n)]
    
    samples_per_group = [0]*n
    for value in x:
        for i, (lower, upper) in enumerate(groups_boundaries):
            if lower <= value < upper:
                samples_per_group[i] += 1
                break
    if n%2 ==0:
        if (samples_per_group[n//2 - 1] > samples_per_group[n//2]):
            mid_index = n//2 - 1
        else:
            mid_index = n//2
    else:
        mid_index = n//2
        
    target_cof_value_gaussian = calculate_cdf(x, x_min + (mid_index+1)*step) - calculate_cdf(x, x_min + mid_index*step)
    target_ratio_gaussian = samples_per_group[mid_index] / target_cof_value_gaussian

    resampled_counts_gaussian = [math.floor(target_ratio_gaussian * ((calculate_cdf(x, x_min + (i+1)*step) - calculate_cdf(x, x_min + i*step)))) for i in range(n)]
    resampled_counts_mean = [max(samples_per_group)] * n  # Simplified mean distribution sampling strategy
    
    df_gaussian = pd.DataFrame()  # Initialize result DataFrame
    for idx, (a, b) in enumerate(zip(samples_per_group, resampled_counts_gaussian)):
        # Determine start and end indices for slicing the DataFrame
        start_idx = sum(samples_per_group[:idx])
        end_idx = sum(samples_per_group[:idx+1])

        if b > a:
            # Additional Samples Required
            additional_samples = b - a
            # Draw additional samples from the current group (with replacement)
            extra_samples = df.iloc[slice(start_idx, end_idx)].sample(n=additional_samples, replace=True, random_state=random_state)
            # Concatenate all original samples with the additional samples within the group
            df_gaussian_temp = pd.concat([df.iloc[slice(start_idx, end_idx)], extra_samples], ignore_index=True)
        else:  
            # Draw the required number of samples directly from the current group
            sampled_df = df.iloc[slice(start_idx, end_idx)].sample(n=b, replace=False, random_state=random_state)
            df_gaussian_temp = sampled_df

        # Append the processed data from the current group to the final DataFrame
        df_gaussian = pd.concat([df_gaussian, df_gaussian_temp], ignore_index=False)

    sample_indices = df_gaussian.index
    remaining_indices = ~df.index.isin(sample_indices)
    remaining_df = df.loc[remaining_indices]
        
    return df_gaussian.reset_index(drop=True), remaining_df.reset_index(drop=True)

def stratified_sample_train_test(df: pd.DataFrame, n: int, ratio: float, random_state: int = 42) -> tuple:
    """
    Stratify the input DataFrame based on the last column (label), divide into n groups,
    then sample within each group according to the specified ratio to form a training set.
    The remaining data forms the test set.
    
    Parameters:
    df (pd.DataFrame): The original DataFrame with the last column being the target variable.
    n (int): Number of groups to divide the data into.
    ratio (float): Ratio of samples to take from each group for the training set.
    random_state (int, optional): Seed for reproducible sampling. Defaults to 42.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Training DataFrame and Test DataFrame.
    """
    # Sort DataFrame by the label column
    df_sorted = df.sort_values(by=df.columns[-1], ascending=True)
    
    # Divide data into n groups and calculate sample counts per group
    x = df_sorted.iloc[:, -1].values
    x_min, x_max = x.min(), x.max()
    step = (x_max - x_min) / n
    groups_boundaries = [(x_min + i*step, x_min + (i+1)*step) for i in range(n)]
    samples_per_group = [0]*n
    for value in x:
        for i, (lower, upper) in enumerate(groups_boundaries):
            if lower <= value < upper:
                samples_per_group[i] += 1
                break
                
    # Sample from each group and construct the training set
    df_train = pd.DataFrame()
    for idx, count in enumerate(samples_per_group):
        start_idx = sum(samples_per_group[:idx])
        end_idx = sum(samples_per_group[:idx+1])
        resample_count = int(math.ceil(count * ratio))
        df_train_temp = df_sorted.iloc[start_idx:end_idx].sample(n=resample_count, replace=False, random_state=random_state)
        df_train = pd.concat([df_train, df_train_temp], ignore_index=False)
        
    # Construct the test set with the remaining data
    sample_indices = df_train.index
    remaining_indices = ~df.index.isin(sample_indices)
    df_test = df.loc[remaining_indices]
    
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)