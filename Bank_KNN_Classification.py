"""
Created on Sat Oct 21 16:44:59 2023

@author: Ivan Sapozhnikov
Fall 2023
DS 2500 
HW 3
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

BANK_FILE_PATH = 'banklist.csv'
INSTITUTION_FILE_PATH = 'institutions.csv'
COLUMNS_TO_NORMALIZE = \
    ["ASSET", "DEP", "DEPDOM", "NETINC", "OFFDOM", "ROA", "ROAPTX", "ROE"]
LOWEST_K_TO_CHECK = 4
HIGHEST_K_TO_CHECK = 18
PRED_BANK_NAME = "Southern Community Bank"
PRED_BANK_TOWN = "Fayetteville"
PRED_BANK_ST_ABRV = "GA"

def normalize_column(df, column_name):
    """
    Min-Max Normalizes the values in a specified column of a DataFrame
    Adds a "orginal column name"_norm column with the normalized data
    
    Parameters:
    df: Pandas DataFrame with column to be normalized
    column_name (str): The name of the column in the DataFrame to be normalized
    
    Returns:
    None: Modifies the input DataFrame 
    """
    col_min = df[column_name].min()
    col_max = df[column_name].max()
    
    # Perform min-max normalization and assign the result to a new column
    df[column_name + "_norm"] = (df[column_name] - col_min)/(col_max - col_min)
    
def remove_empty_rows(df, subset):
    """
    Removes rows with missing values in specified columns
    Resets the index of the DataFrame.
    
    Parameters:
    df: The DataFrame to remove empty rows from
    subset (array-like): A list of columns to check for missing values

    Returns:
    None: Function modifies the DataFrame in place.
    """
    
    # Remove rows with NaN values in the specified columns
    df.dropna(subset=subset, inplace=True)
    
    # Reset the index of the DataFrame
    df.reset_index(drop=True, inplace=True)
    
def read_in_the_files(bank_file=BANK_FILE_PATH, 
                      institution_file=INSTITUTION_FILE_PATH):
    """
    Load, preprocess, and return a DataFrame
    Specifically: Remove the empty rows and add a column "Did_it_fail" 
    where (1-it failed and 0-it didn't fail) which is created on top of
    of the institution_file and cross validated with the bank_file
    
    Parameters:
    bank_file (str): Path to the CSV file 'banklist.csv'
    institution_file (str): Path to the CSV file 'institutions.csv'
    
    Returns:
    pandas.DataFrame: A preprocessed DataFrame
    """
    # Load the CSV files into DataFrames
    failed_bank_df = pd.read_csv(bank_file, encoding='cp1252')
    banks_df = pd.read_csv(institution_file, low_memory=False)

    # Flag whether a bank has failed with 1 (failed) 0 (didn't fail)
    banks_df["Did_it_fail"] = \
            np.where(banks_df['CERT'].isin(failed_bank_df['Cert ']), 1, 0)
    
    # Remove rows with missing values in specified columns
    remove_empty_rows(banks_df, COLUMNS_TO_NORMALIZE)
    
    # Normalize specified columns in the DataFrame
    for col_name in COLUMNS_TO_NORMALIZE:
        normalize_column(banks_df, col_name)
    
    return banks_df

def get_xy_and_train(df, x_cols, y_col_name="Did_it_fail"):
    """
    Prepare datasets by split into training and testing sets.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame with data
    x_cols (list): List of column names to be used as features.
    y_col_name (str, Default: "Did_it_fail"): Name of the target column.
        
    Returns:
    list: A list with features (x), target (y), and training and testing sets
          x, y, x_train, x_test, y_train, y_test
    """
    x = df[x_cols]  # Select feature columns
    y = df[y_col_name]  # Select the target column
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    
    # Export the x_y_train_test_list which other functions know how to unpack
    x_y_train_test_list = [x, y, x_train, x_test, y_train, y_test]
    
    return x_y_train_test_list

def max_min_col_give_k(df, column_to_analyze, k_column="k"):
    """
    Find the min and max values in a specified column
    Return corresponding values from another column and
        the min max of the given column
    
    Parameters:
    df: The DataFrame to analyze.
    column_to_analyze (str): Column name to find the min and max values 
    (ex. 'Accuracy', 'Precision', 'Recall', 'Predicted_Ys', 'F1_scores')
    k_column (str, Default: "k"): Name of the column to retrieve values from
    
    Returns: corresponding values from k_column and actual min and max values 
    from column_to_analyze.
    """
    # Finding the rows where min and max values are in column_to_analyze
    min_row = df[df[column_to_analyze] == df[column_to_analyze].min()]
    max_row = df[df[column_to_analyze] == df[column_to_analyze].max()]
    
    # Getting the values from k_column in the same rows as min and max values
    min_k = min_row[k_column].values[0]
    max_k = max_row[k_column].values[0]
    
    # Getting the actual min and max values from column_to_analyze
    min_value = min_row[column_to_analyze].values[0]
    max_value = max_row[column_to_analyze].values[0]
    
    return min_k, min_value, max_k, max_value

def train_and_evaluate_knn(x, y, x_train, y_train, k):
    """
    Train a k-NN classifier and calculate performance different metrics
    Used within train_and_evaluate_knn
    Parameters:
    x (DataFrame): The featurest for the data.
    y (List): The target variable for all data.
    x_train (DataFrame): The features for the training data.
    y_train (List): The target variable for the training data.
    k (int): The number of neighbors to use for k-NN.
    Returns: A tuple with pred x, f1 score, accuracy, precision, and recall
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn = knn.fit(x_train, y_train)
    cv = KFold(n_splits=4, random_state=0, shuffle=True)

    # Predicted y and F1 score calculation
    y_pred = knn.predict(x)
    xtrain_pred = knn.predict(x_train)
    f1 = f1_score(y_train, xtrain_pred, zero_division=0)
        
    # Calculation all the other metrics: Accuracy, Precision and Recall
    metrics = [accuracy_score, precision_score, recall_score]
    scores = []
    for metric in metrics:
        score = np.mean(cross_val_score(knn, x, y, 
                                        cv=cv, scoring=make_scorer(metric)))
        scores.append(score)
    accuracy, precision, recall = scores
    return y_pred, f1, accuracy, precision, recall

def knn_performance_over_k_values(x_y_train_test_list, k_values):
    """
    Evaluate the k-NN classifier performance over a range of k values
    compile the performance metrics into a DataFrame.

    Parameters:
    x_y_train_test_list (list): A list with necessary feature combinations
    k_values (range): A range of k values to evaluate.
    
    Returns:
    DataFrame: A DataFrame with the performance metrics for each value of k
    """
    # Unpack x_y_train_test_list into x, y and train/test sets
    x, y, x_train, x_test, y_train, y_test = x_y_train_test_list
    
    # The column names for the DataFrame for storing performance metrics
    columns = \
        ['k', 'Accuracy', 'Precision', 'Recall', 'Predicted_Ys', 'F1_scores']
    k_acc_df = pd.DataFrame(columns=columns)
    
    for i, k in enumerate(k_values):
        y_pred, f1, accuracy, precision, recall = \
        train_and_evaluate_knn(x, y, x_train, y_train, k)
        k_acc_df.loc[i] = [k, accuracy, precision, recall, y_pred, f1]
    return k_acc_df

def create_confusion_matrix(x_y_train_test_list, k):
    """
    Create a confusion matrix
    
    Parameters:
    x_y_train_test_list (list): A list with necessary feature combinations
    k (int): The number of neighbors to use for k-NN.

    Returns:
    numpy.ndarray: A confusion matrix of predicted versus actual values.
    """
    # Unpack x_y_train_test_list into x, y and train/test sets
    x, y, x_train, x_test, y_train, y_test = x_y_train_test_list

    
    # Creating and training the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    # Making predictions on the test set
    y_pred = knn.predict(x_test)
    
    # Generating the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return conf_matrix

def calculate_f1_from_confusion_matrix(conf_matrix):
    """
    Calculate the F1 score from the given confusion matrix
    
    Parameters:
    conf_matrix (numpy.ndarray): A 2x2 confusion matrix
    
    Returns:
    float: The F1 score computed from the confusion matrix
    """
    TP = conf_matrix[1, 1]  # True Positives
    FP = conf_matrix[0, 1]  # False Positives
    FN = conf_matrix[1, 0]  # False Negatives
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    # The formula for f1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

def predict_specific_bank(banks_df,
        bank_name, city, state, x_y_train_test_list, x_col, k):
    """
    Predict whether a specific bank failed or not 
    Compare it with the actual outcome.
    """
    # Unpack x_y_train_test_list into x, y and train/test sets
    x, y, x_train, x_test, y_train, y_test = x_y_train_test_list
    
    # Find the row for the specific bank
    bank_row = banks_df[(banks_df['NAME'] == bank_name) & 
                        (banks_df['CITY'] == city) & 
                        (banks_df['STALP'] == state)]
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    
    # Making a prediction for the specific bank
    bank_features = bank_row[x_col].values.reshape(1, -1)
    prediction = knn.predict(bank_features)
    
    # Getting the actual outcome for the specific bank
    actual_outcome = bank_row['Did_it_fail'].values[0]
    
    return prediction[0], actual_outcome

def plot_heatmap(conf_matrix):
    '''
    Plot a heatmap for the confusion matrix of a k-NN predictions.
    '''
    sns.heatmap(conf_matrix, 
        annot=True, fmt="d", cmap="Greens", 
        xticklabels=["Not Failed", "Failed"], 
        yticklabels=["Not Failed", "Failed"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('K-NN Confusion Matrix Heatmap')
    plt.show()

def plotting_preformance_over_k(df):
    """
    Plot the performance metrics (Accuracy, Precision, Recall) over k values
    Parameter: df (DataFrame) A DataFrame with the k values and metrics.
    """
    # Plotting Accuracy, Precision, and Recall line plots
    plt.figure(figsize=(11, 7))
    plt.plot(df['k'], df['Accuracy'], label='Accuracy', color='green')
    plt.plot(df['k'], df['Precision'], label='Precision',  color='blue')
    plt.plot(df['k'], df['Recall'], label='Recall', color='purple')

    # Highlighting the k values with the highest recall and accuacy
    optimal_recall_k = df.loc[df['Recall'].idxmax(), 'k']
    optimal_recall = df['Recall'].max()
    plt.scatter(optimal_recall_k, optimal_recall, color='black',
                zorder=2, label = "Optimal Recall")
    optimal_accuracy_k = df.loc[df['Accuracy'].idxmax(), 'k']
    optimal_accuracy = df['Accuracy'].max()
    plt.scatter(optimal_accuracy_k, optimal_accuracy, color='red',
                zorder=2, label = "Optimal Accuracy")

    # Labeling
    plt.title('k-NN Performance Metrics vs. k Values')
    plt.xlabel('k Value')
    plt.ylabel('Metric Score')
    plt.xticks(df['k'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Preprocessing of the data
    banks_df = read_in_the_files( \
        bank_file = BANK_FILE_PATH, institution_file = INSTITUTION_FILE_PATH)
    norm_columns = [col + "_norm" for col in COLUMNS_TO_NORMALIZE]
    x_y_train_test_list = get_xy_and_train( \
        df = banks_df, x_cols = norm_columns, y_col_name = "Did_it_fail")

    # Evaluate the model for questions 1 through 4
    k_values = range(LOWEST_K_TO_CHECK, HIGHEST_K_TO_CHECK +1) # range val of k
    k_acc_df = knn_performance_over_k_values(x_y_train_test_list, k_values)
    
    # Calculate min and max k values and scores for Precision Accuracy & Recall
    ac_min_k, ac_min_value, ac_max_k, ac_max_value = \
        max_min_col_give_k(k_acc_df, "Accuracy", k_column = "k")
    pr_min_k, pr_min_value, pr_max_k, pr_max_value = \
        max_min_col_give_k(k_acc_df, "Precision", k_column = "k")
    re_min_k, re_min_value, re_max_k, re_max_value = \
        max_min_col_give_k(k_acc_df, "Recall", k_column = "k")

    # Question 1_______________________________________________________________
    # What is the optimal value of k if we care most about accuracy?
    print("The optimal value of k if we care most about accuracy is", ac_max_k)
    
    # Question 2_______________________________________________________________
    # What is the lowest mean accuracy for any value of k?
    print(f"The lowest mean accuracy for any value of k is {ac_min_value}")
    
    # Question 3_______________________________________________________________
    # What is the optimal value of k if we care most about overall precision?
    print("The optimal value of k if we care most about overall precision is",
                                                                      pr_max_k)
    # Question 4_______________________________________________________________
    # What is the optimal value of k if we care most about overall  recall?
    print("The optimal value of k if we care most about overall  recall is",
                                                                      re_max_k)
    # Get the confusion matrix for the data (to be used to answer 5 a & b)
    conf_matrix = create_confusion_matrix(x_y_train_test_list, ac_max_k)
    
    # Question 5a______________________________________________________________
    # Let’s say we set the value of k to the optimal value
    # if we care most about accuracy. Then..
    f1_failed_banks = calculate_f1_from_confusion_matrix(conf_matrix)
    print("The f1 score for just one class -- the banks that failed is",
                                                               f1_failed_banks)
    
    # Question 5b______________________________________________________________
    # How many banks did your model predict to NOT fail, and in fact did not?
    true_neg = conf_matrix[0, 0]
    print("The number of banks my model predicted to NOT fail,",
                                            "and in fact did not is", true_neg)

    # Question 5c______________________________________________________________
    # Does your model correctly predict what happened to 
    # Southern Community Bank of Fayetteville, GA?
    prediction, actual_outcome = \
        predict_specific_bank(banks_df, PRED_BANK_NAME, PRED_BANK_TOWN, 
        PRED_BANK_ST_ABRV, x_y_train_test_list, norm_columns, k=ac_max_k)
    print("Our model predicts that the bank will")
    if prediction == 1:
        print("FAIL")
    elif prediction == 0:
        print("NOT FAIL")
    print("And the actual outcome for the bank is that it")
    if prediction == 1:
        print("FAILED")
    elif prediction == 0:
        print("DID NOT FAIL")

    # PLOTTING QUESTIONS_______________________________________________________
    # Plot #1: A heatmap showing the confusion matrix
    # when the value of k is optimal if we care most about recall.
    plot_heatmap(conf_matrix)

    # Plot #2: A plot showing why you picked those optimal values of k. 
    # You can use more than one plot here, or subplots, as long as 
    # we can see and understand how different values of k correlate with
    # accuracy, precision, and recall.
    plotting_preformance_over_k(k_acc_df)

main()