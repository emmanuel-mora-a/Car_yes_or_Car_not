# Final Project for Data science Graduation from Stackfuel Gmbh

Participants receive another larger dataset that they must analyze independently, with less guidance compared to the practice project. In an individual project discussion with the mentoring team from StackFuel, participants receive feedback on their approach to the solution.

# Why the notebook contains screenshot for the result

Regrettably, Stackfuel GmbH is unable to permit the release of our learning materials or completed projects due to copyright regulations and associated licensing rights. To showcase my project externally, I would need to recode everything outside the learning environment. However, without access to the dataset, the results can only be viewed within the learning environment. Consequently, the only method to display my work externally is through screenshots

# Scenario

You work as a data scientist for a US used car dealer. The dealer buys used cars at low prices in online auctions and from other car dealers in order to resell them profitably on their own platform. It's not always easy to tell whether it is worth buying a used car: One of the biggest challenges in used car auctions is the risk of a car having problems that are so serious, that they prevent it from being resold to customers. These are referred to as "lemons" - cars that have significant defects from the outset due to production faults that significantly affect the safety, use or value of that car and at the same time cannot be repaired with a reasonable number of repairs or within a certain period of time. In cases like this, the customer has the right to be refunded the purchase price. In addition to the purchase costs, the bad purchase of these so-called lemons leads to considerable costs as a result, such as the storage and repair of the car, which can result in market losses when the vehicle is resold.

That is why it is important for your boss to rule out as many bad purchases as possible. To help the buyers in the company with the large number of cars on offer, you are to develop a model that predicts whether a car would be a bad buy, a so-called lemon. However, this must not lead to too many good purchases being excluded. You won't receive more detailed information on the costs and profits of the respective purchases for developing the prototype just yet.

### ***What problem should the model solve?***

The model should aim to predict whether a given used car is likely to be a "lemon" (a car with significant defects) that would result in a financial loss for the company. By identifying these potentially problematic cars before purchase, the model helps to minimize financial risks and ensures that only cars in good condition are acquired.

---

### ***What is the nature of the problem?***

This is a classification problem. The task is to classify each car into one of two categories:

Lemon (problematic car).

Not a Lemon (non-problematic car).

---

### ***What would an application using your model look like?***

 **Application Appearance and Functionality:**

**User Interface:** The application could have a web-based or mobile interface accessible by company buyers.

**Input Fields:** Buyers can input various features of a car (e.g., make, model, year, mileage, service history, etc.).

**Prediction Output:** The application processes the input data and returns a prediction indicating whether the car is likely to be a "lemon".

**Recommendation System:** Provide insights or recommendations based on the prediction, such as advising against the purchase or suggesting further inspection.

---

### ***Requirements Specified by Employer:***

**Accuracy and Recall:** The model must effectively distinguish between lemons and non-problematic cars while minimizing the exclusion of good purchases.

**Ease of Use:** The application should be user-friendly and provide clear, actionable insights.

**Scalability:** The model should handle a large volume of data and predictions efficiently.

**Explainability:** The model should provide some level of interpretability or explainability to help users understand the reasons behind each prediction.

---

### ***What data do you need so that you can build your model?***

**To build an effective model, the following data is required:**

**Car Features:**

Make and model.

Year of manufacture.

Mileage.

Service and repair history.

Previous ownership details.

Condition reports.

Auction or seller details.

**Historical Data:**

Records of past car purchases, including those identified as lemons and non-problematic cars.

Detailed description of the issues faced by lemons.

Financial data on purchase and repair costs, as well as resale values.

**Label Data:**

Binary labels indicating whether each car in the historical dataset was a lemon.

**External Data:**

Industry reports and statistics on common defects and their occurrence in specific car models or makes.

Customer feedback or reviews that might indicate recurring issues

# Preparation

##### import modules

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

##### useful imports

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  # Corrected the typo here
from sklearn.svm import SVC, LinearSVC  # Corrected the capitalization here
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  # Corrected the typo here
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, classification_report  # Corrected the typo here
import os
import pickle as p
from tensorflow.keras.models import load_model  # Corrected the import here
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

from sklearn.cluster import DBSCAN
from sklearn.metrics import euclidean_distances

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import warnings  # Corrected the typo here
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

plt.style.use('ggplot')

# Importing data + EDA

Exploring the data through these steps is essential to understand its structure, quality, and underlying patterns. This process helps identify potential issues that need to be addressed before building models or conducting deeper analysis, ensuring the accuracy and reliability of your results.

#### Loading the Dataset

Loading the dataset is the first step to any analysis. It allows you to work with the data within your programming environment.

#### Converting the Purchase Date

Converting timestamps to a datetime format makes it easier to handle and analyze date-related data, such as calculating the time between events or aggregating data by month or year.

#### Displaying the DataFrame Dimensions

Knowing the dimensions of your dataset (number of rows and columns) gives you an initial understanding of its size and scope, which is essential for planning further analysis.

#### Descriptive Statistics

Descriptive statistics provide a summary of the central tendency, dispersion, and shape of the dataset’s distribution. This helps identify outliers, trends, and potential issues in the data

#### DataFrame Information

The `info()` method provides a concise summary of the DataFrame, including data types, non-null values, and memory usage. This is important for identifying columns with missing values and ensuring that data types are appropriate for analysis

#### Checking for Missing Values

Identifying missing values is crucial as they can skew analysis results and model performance. Knowing which columns have missing values allows you to decide how to handle them, whether through imputation, removal, or another method

#### Checking for Duplicate Rows

Duplicates can distort your analysis and lead to biased results. Detecting and handling duplicate rows ensures that each data point is unique and accurately represented in your analysis

#### Function to Add Spaces Between Outputs

Adding spaces between outputs improves readability, making it easier to review and interpret the printed results

# Train - Test - Split

Performing a train-test split is a critical step in machine learning. It ensures that you have separate datasets for training and evaluating your model, allowing you to assess how well your model generalizes to new, unseen data. Here’s a brief summary of the steps:

- **Step 1**: Separate the dataset into feature variables and target variable.

- **Step 2**: Use `train_test_split` to divide the data into training and testing sets.
  
  - **Training Set**: Used to train the model.
  
  - **Testing Set**: Used to test the model’s performanc

### Performing Train-Test Split

The dataset is split into two parts: features (`df_features`) and target (`df_target`).

- `df_features`: Contains all columns except `IsBadBuy`, which are the input variables (features) used to predict the target.

- `df_target`: Contains the `IsBadBuy` column, which is the output variable (target) the model aims to predict.

### Train-Test Split:

The `train_test_split` function from `sklearn.model_selection` is used to split the dataset into training and testing sets.

- `features_train` **and** `target_train`: These are the training sets used to train the machine learning model.

- `features_test` **and** `target_test`: These are the testing sets used to evaluate the model’s performance.

- `test_size=0.1`: Specifies that 10% of the data should be used for testing, and the remaining 90% for training.

- `random_state=42`: Ensures reproducibility of the split by setting a fixed random seed. This means that every time you run the code, the split will be the same.

# Data Preparation + Working with features_train

##### Getting cat and num colunms from features_train

Separation of Categorical cols, and numeric cols for futher exploration

##### Getting to know how many unique values each col feature has

This part of the code is essential for performing an initial exploration of the categorical columns in your dataset. Here’s why each step is important:

- **Identifying Unique Values and Counts**: Helps understand the distribution and frequency of categories in the data.

- **Verifying Data Types**: Ensures that the columns have the correct data types for categorical data.

- **Readable Output**: Organizes the output for better readability, making it easier to interpret the results.

By analyzing the unique values, their counts, and the data types of the categorical columns, you gain insights into the structure and quality of your categorical data. This information is crucial for further preprocessing and modeling steps

##### Looking at histogram of numeric cols

This part of the code creates a grid of histograms for each numerical column in the training features set (`features_train`). Here’s why each step is important:

- **Grid Setup**: Determines the layout of the subplots, ensuring all histograms fit neatly into the grid.

- **Subplot Creation**: Generates the figure and subplots for visualizing the histograms.

- **Histogram Plotting**: Plots histograms for each numerical column, providing a visual distribution of the data. This helps in understanding the distribution and detecting any potential issues such as skewness, outliers, or gaps in the data.

Creating these histograms is crucial for exploratory data analysis (EDA), as it allows you to visually inspect the distribution of numerical features and gain insights into the underlying patterns of the data

![](C:\Users\emman\AppData\Roaming\marktext\images\2024-11-30-23-14-02-image.png)

here i started identifiying some outliers

# Cleaning the Nan Values

#### Data_cleaner function

This `data_cleaner` function is a comprehensive data cleaning and preprocessing tool. It performs the following key actions:

- Drops unnecessary columns.

- Imputes missing values with appropriate references from the training set.

- Ensures consistent data types for numerical and categorical columns.

By cleaning and standardizing the dataset, this function helps to improve the quality of the data and enhances the reliability and accuracy of any subsequent analysis or machine learning models

the NaN in categorail features will be replace with the mode 

the NaN in numerical features will be replace with the median 

# Deal with outliers

### funtion iqr_cleaner

The `iqr_cleaner` function provides a visual way to identify and inspect outliers in your numerical data by:

- Setting up a grid of scatter plots.

- Iterating through numerical columns to calculate IQR and identify outliers.

- Creating scatter plots with outliers highlighted.

This visualization helps to understand the distribution of numerical features and detect any extreme values that may need to be addressed in the data preprocessing stage.

after exploring the data I found many outliers with 0 value, to create a normal range of values i create a funtion to delete those records outside of some ranges in some columns

testing the function with the current data i get:

![](C:\Users\emman\AppData\Roaming\marktext\images\2024-11-30-23-15-24-image.png)

![](C:\Users\emman\AppData\Roaming\marktext\images\2024-11-30-23-15-55-image.png)

### Funtion Sampling_data

function filters the dataset based on predefined criteria for specific columns. Here’s why each step is important:

- **Filtering Criteria**: The criteria are specified to ensure that the values in certain columns fall within a given range.

- **Applying Filters**: This step iterates through the criteria and filters the DataFrame accordingly, reducing the data to only include rows that meet the conditions.

- **Returning Filtered Data**: The function returns the cleaned and filtered DataFrame, making it ready for further analysis or modeling.

By applying these filters, the function ensures that the dataset is refined and adheres to specific thresholds, which can help in removing outliers or extreme values that might skew the analysis

# Modeling

### Data  New import

This segment of the code is responsible for preparing the data for machine learning by:

- **Loading the Dataset**: Importing data from a CSV file into a DataFrame.

- **Splitting the Data**: Dividing the data into training and testing sets.

- **Cleaning the Data**: Removing unnecessary columns and filling missing values to ensure data quality.

- **Filtering the Data**: Applying additional criteria to refine the training data.

- **Separating Features and Target**: Dividing the data into feature variables and the target variable for both training and testing sets

# Funktion: results_add

This code segment sets up a DataFrame to store the performance metrics of different models and defines a function to add these metrics to the DataFrame. The function evaluates a model on the training, validation, and test sets, calculates various metrics (F1 score, precision, recall, accuracy), and adds these metrics to the results DataFrame.

By doing so, this code provides a systematic way to compare the performance of multiple models across different datasets, making it easier to identify the best-performing model.

# Feature Engineering

#### Sortierung der Spalten

This code segment organizes the columns of the dataset into different groups based on their characteristics and intended processing methods. It then drops the unnecessary columns to create a refined DataFrame for further analysis or modeling. Here's a summary of the steps:

- **Categorical Columns**: Identifies the categorical features for specialized processing.

- **Numerical Columns**: Identifies the numerical features ready for use.

- **Columns for PCA**: Groups columns for dimensionality reduction using PCA.

- **Columns to Drop**: Lists columns that are not needed and should be removed.

- **Dropping Unnecessary Columns**: Reduces the dataset to only the relevant features for analysis or modeling.

By organizing and cleaning the dataset in this way, you ensure that the data is well-prepared for further processing, making it more manageable and effective for machine learning tasks

### One Hot Encoding

This part of the code is focused on transforming and preparing the data for machine learning by:

- **Defining the ColumnTransformer**: Specifies how to transform the categorical columns using One-Hot Encoding.

- **Applying the Transformation**: Applies the defined transformation to the training data.

- **Creating a DataFrame with Transformed Data**: Converts the transformed data into a DataFrame with appropriate column names.

- **Combining Data**: Combines the transformed categorical columns with the original numerical columns to create a final DataFrame ready for model training.

### Funktion: Enginner_features

The `engineer_features` function streamlines the preprocessing and transformation of your data by:

- **Dropping Unnecessary Columns**: Removing columns that are not needed for the analysis.

- **Applying One-Hot Encoding**: Transforming categorical columns into a format suitable for machine learning models.

- **Combining Relevant Columns**: Merging the transformed categorical columns with the numerical columns to create a comprehensive and ready-to-use DataFrame.

This function ensures that the data is well-prepared and standardized, making it more effective for building robust machine learning models

### Scale

This code standardizes the features in your training set to have a mean of 0 and a standard deviation of 1. Here's a summary of the steps:

- **Initializing StandardScaler**: Sets up the scaler to standardize the data.

- **Fitting and Transforming Data**: Learns the mean and standard deviation from the training data, then transforms the data accordingly.

- **Creating Standardized DataFrame**: Converts the standardized data back into a DataFrame with the same structure as the original features.

Standardizing the data is an important step in preprocessing, especially for algorithms that are sensitive to the scale of data, such as gradient descent-based methods. This ensures that all features contribute equally to the model training process

### Dimensionality Reduction

This part of the code performs and visualizes a Principal Component Analysis (PCA) on the columns related to Market Prices (MMR data) to determine how many principal components are necessary to explain the majority of the variance in the data. Here's a step-by-step summary:

- **Selecting MMR Columns**: Focuses the analysis on specific columns related to Market Prices.

- **Initializing Variance Ratio List**: Prepares to store the results of the explained variance ratios.

- **Looping Through PCA Components**: Fits PCA models with varying numbers of components and records the cumulative explained variance ratios.

- **Printing Results**: Outputs the explained variance ratios for each number of components tested.

- **Plotting Variance Ratios**: Visualizes the cumulative explained variance ratios to help determine the optimal number of principal components.

This analysis helps in understanding the dimensionality of the data and selecting an appropriate number of components for PCA, ensuring that the most important features are retained while reducing dimensionality

### PCA

This code segment performs Principal Component Analysis (PCA) on specific sets of columns and integrates the transformed data with the original features. Here’s a step-by-step summary:

- **Define PCA Column Names**: Specifies the names for the new columns created by PCA.

- **Initialize ColumnTransformer for PCA**: Sets up the transformation to apply PCA to selected columns.

- **Fit and Transform Data**: Fits the transformer to the data and applies the PCA transformation.

- **Create DataFrame for PCA Components**: Converts the transformed data into a DataFrame with appropriate column names.

- **Combine with Original Data**: Merges the PCA components with the original standardized data.

- **Drop Original PCA Columns**: Removes the original columns used for PCA to avoid redundancy.

### Component Composition

This visualization helps to interpret how the original features contribute to each principal component, making it easier to understand the relationships between features and the PCA-transformed data

### Function scale_features

This function scales the features of a given DataFrame using a specified scaler.

- **Inputs**:
  
  - `df`: The DataFrame containing the features to be scaled.
  - `scaler`: An optional parameter, which is a pre-fitted scaler object (e.g., from `sklearn.preprocessing`) that standardizes or normalizes the data.

- **Functionality**:
  
  - Creates a copy of the input DataFrame to avoid modifying the original data.
  - Transforms the features using the provided scaler.
  - Constructs a new DataFrame from the transformed data, preserving the original columns and index.
  - Returns the scaled DataFrame.

**Purpose**: This function ensures that feature scaling is consistently applied, which is essential for many machine learning models to perform optimally.

### Funtion pca_features

This function applies Principal Component Analysis (PCA) transformation to a DataFrame, replacing specified original features with the new PCA-transformed features.

- **Inputs**:
  
  - `df`: The original DataFrame containing features to be transformed.
  - `cols_pca_new`: The names of the new columns for the PCA-transformed features (default is `cols_pca`).
  - `transformer_pca`: A pre-fitted PCA transformer object used to perform the transformation.
  - `cols_pca_old`: A list of the original feature columns to be replaced, which defaults to `cols_pca_MMR + cols_pca_veh_age_year`.

- **Functionality**:
  
  - Creates a copy of the input DataFrame to preserve the original data.
  - Uses the PCA transformer to transform the original features.
  - Constructs a new DataFrame with the PCA-transformed features (`df_copy_MMR`) and appends it to the original DataFrame.
  - Drops the old feature columns specified in `cols_pca_old` from the DataFrame.
  - Returns the modified DataFrame, now containing the new PCA features in place of the original ones.

**Purpose**: This function integrates PCA into the feature engineering process, reducing dimensionality and potentially enhancing model performance by transforming correlated features into a set of uncorrelated principal components

# Create Validation Set

This section of code performs data splitting to create training and validation sets and then saves these datasets as Pickle files for efficient future use. Here’s what each part does:

1. **Splitting Data for Validation**:
   
   - `train_test_split(...)`: This function splits the data into training and validation sets, using a 70-30% split (controlled by `test_size=0.3`) and a fixed `random_state` to ensure reproducibility.
   - The first `train_test_split` separates the data for the original set of engineered features.
   - The second `train_test_split` separates the data for a different set of engineered features, which include polynomial features (indicated by the `_poly` suffix).

2. **Saving Data as Pickle Files**:
   
   - `to_pickle(...)`: This method saves the split data to Pickle files. Pickle is a serialization format in Python that allows quick saving and loading of data structures.
   - The original feature sets (`features_train_engineered_ready` and `features_val_engineered_ready`), along with their corresponding target variables, are saved as Pickle files.
   - The datasets for the first set of selected features, including polynomial transformations, are also saved.

**Purpose**:

- **Data Splitting**: Creating separate training and validation sets is crucial for model evaluation and preventing overfitting.
- **Pickle Saving**: Saving the data helps avoid re-running potentially time-consuming data preparation processes, making future model training and evaluation faster and more efficient.

# Load Dataset

his section of code loads pre-saved datasets from Pickle files for quick and efficient access. Here's what each part does:

1. **Loading Original Engineered Features**:
   
   - `pd.read_pickle(...)`: This function reads the serialized Pickle files and loads them into DataFrames.
   - The original training, validation, and test sets of engineered features (`features_train_engineered_ready`, `features_val_engineered_ready`, `features_test_engineered_ready`) are loaded.
   - The corresponding target variables (`target_train`, `target_val`, `target_test`) are also loaded.

2. **Loading First Selection of Features**:
   
   - The first selection of engineered features, which may include additional transformations such as polynomial features, is loaded into DataFrames (`features_train_engineered_ready_poly`, `features_val_engineered_ready_poly`, `features_test_engineered_ready_poly`).

3. **Assigning Target Variables**:
   
   - The target variables for the polynomial feature set (`target_train_poly`, `target_val_poly`, `target_test_poly`) are set to be the same as the original target variables, since they share the same labels.

**Purpose**: This code is designed to quickly load prepared datasets for training and evaluation, making the workflow efficient by avoiding repetitive data processing. The use of Pickle files ensures that the data structures are easily restored for model training and testing

# Resampling

1. **Initializing Samplers**:
   
   - `SMOTE`, `RandomOverSampler`, and `RandomUnderSampler` are sampling techniques from the `imbalanced-learn` library.
     - `SMOTE (Synthetic Minority Over-sampling Technique)`: Generates synthetic samples for the minority class to balance the dataset.
     - `RandomOverSampler`: Randomly duplicates samples from the minority class.
     - `RandomUnderSampler`: Randomly removes samples from the majority class to achieve balance.
   - All samplers are initialized with a `random_state` for reproducibility.

2. **Applying Samplers to Original Features**:
   
   - `fit_resample(...)`: This method applies the sampling techniques to the original set of engineered features (`features_train_engineered_ready`) and the corresponding target (`target_train`).
   - The result is three balanced datasets:
     - `features_train_engineered_ready_SMOTE` and `target_train_SMOTE`: Balanced using SMOTE.
     - `features_train_engineered_ready_Over` and `target_train_Over`: Balanced using random oversampling.
     - `features_train_engineered_ready_Under` and `target_train_Under`: Balanced using random undersampling.

3. **Applying Samplers to the First Selection of Features**:
   
   - The same sampling techniques are applied to the first set of selected features (potentially including polynomial features).
   - The resulting datasets are similarly balanced:
     - `features_train_engineered_ready_poly_SMOTE` and `target_train_poly_SMOTE`: Balanced using SMOTE.
     - `features_train_engineered_ready_poly_Over` and `target_train_poly_Over`: Balanced using random oversampling.
     - `features_train_engineered_ready_poly_Under` and `target_train_poly_Under`: Balanced using random undersampling.

**Purpose**: Balancing the training data is crucial for training machine learning models, especially when the classes are imbalanced. Different sampling techniques provide various ways to address this issue, allowing for experimentation and comparison to find the best approach for the model.

# Model Testing

In the model testing phase, I tested different model with different parameter to know which one explains the data at best and also give the best results

next models got tested:

RandomForestClassifier

LogisticRegression

KNeighborsClassifier

SVC  Support Vector Classifier

Artificial Neural Networks

# Model Selection

with the use of the function ***results_add*** with eacf of the tested models i get the next dataframe with all of the results:

<img src="file:///C:/Users/emman/AppData/Roaming/marktext/images/2024-11-18-20-07-04-image.png" title="" alt="" width="1006">

# Final model

based on performace  F1 score, i choose the model  **model_RF_1**

# The final Datenpipeline

This function**** ** data_pred******  and processes a given DataFrame for prediction using a pre-trained machine learning model.

- **Inputs**:
  
  - `df`: The input DataFrame containing raw data to be used for predictions.
  - `model`: The trained model used to make predictions. By default, it uses `model_final`.

- **Functionality**:
  
  1. **Convert Date Column**:
     
     - `df.loc[:, "PurchDate"] = pd.to_datetime(...)`: Converts the `"PurchDate"` column from Unix timestamp format to a proper `datetime` format. This step ensures that the date information is correctly formatted for any feature engineering steps.
  
  2. **Data Cleaning**:
     
     - `df = clean_data(df)`: Cleans the data using a custom `clean_data` function. This could involve handling missing values, correcting data inconsistencies, or other preprocessing steps.
  
  3. **Feature Engineering**:
     
     - `df = engineer_features(df)`: Applies feature engineering transformations, such as creating new features or encoding categorical variables.
     - `df = scale_features(df)`: Scales the features using a previously defined `scale_features` function to ensure that all features have a consistent scale, which is essential for model performance.
     - `df = pca_features(df)`: Applies Principal Component Analysis (PCA) to reduce the dimensionality of the features and create a set of uncorrelated principal components.
     - `df = poly_features(df)`: Applies polynomial feature transformations, which can capture non-linear relationships in the data.
  
  4. **Model Prediction**:
     
     - `model.predict(df)`: Uses the pre-trained `model_final` to make predictions on the processed data.

- **Returns**:
  
  - The function returns the predictions made by the model on the processed DataFrame.

**Purpose**: The `data_pred` function automates the entire data preparation process—from cleaning and feature engineering to scaling and dimensionality reduction—before using a machine learning model to make predictions. This ensures consistency and efficiency when making predictions on new data

# 

# completion of the project

This code performs predictions on a new dataset and saves the results to a CSV file.

1. **Loading the New Data**:
   
   - `features_aim = pd.read_csv("features_aim.csv")`: Loads a new dataset from a CSV file named `"features_aim.csv"` into a DataFrame called `features_aim`. This dataset contains the features on which predictions will be made.

2. **Making Predictions**:
   
   - `data_pred(features_aim)`: Calls the `data_pred` function, which processes the `features_aim` DataFrame and uses a pre-trained model to make predictions.
   - `target_aim_pred = pd.DataFrame(..., columns=["IsBadBuy_pred"])`: The predictions are converted into a DataFrame with a single column named `"IsBadBuy_pred"`, making the results easy to work with and interpret.

3. **Saving Predictions to a CSV File**:
   
   - `target_aim_pred.to_csv("predictions_aim.csv")`: Saves the predictions to a CSV file named `"predictions_aim.csv"`, making the results easily shareable or usable for further analysis.

**Purpose**: This code efficiently loads new data, uses a pre-trained model to make predictions, and saves the output to a CSV file for reporting or further use
