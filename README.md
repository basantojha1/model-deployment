# Project Report

# Predicting Hit Song Using Repeated Chorus Project

For this project, we collected data from the billboard Hot 100 website using billboard.py.
Billboard is a website that contains a list of songs with their artist name and other detail which is updated on weekly basis.
In this project, we collect song data form both billboard hot 100 and billboard website and label them as popular and unpopular songs. And, use this data as a base for create a predictive model to predict whether the song is popular or unpopular.

# Understanding the Dataset

**Data Collection and Chorus Extraction**
The dataset we are working on contains total 10 years of data from both billboard hot 100 and billboard global 200. We extracted Song, Artist and Label of the song.
After acquiring 10 years of data from billboard hot 100 songs, I got a total of 1000 songs. The data contained many duplicate values so I removed the duplicate values and finally got 839 songs data. I labeled them as popular songs. 
After acquiring 10 years of data from billboard global 200 songs, I got a total of 2000 songs. The data contained many duplicate values so I removed the duplicate values and finally got 451 songs data after removing 1549 duplicates. I labeled them as unpopular songs. 
Applied filter while concatenating the hot 100 songs dataset with global 200 dataset. The filter concatenated the songs of the artists that are present in the hot 100 dataframe.
The final dataframe was exported in the csv file.
Youtube-search-python package was installed to get the urls of the songs from youtube and added the urls in the csv file.
The songs were downloaded using the urls in .mp3 format, renamed and added in a new downloaded_songs directory.
Ffmpeg-python package was installed and pydub was also installed in order to get started with chorus extraction. A new chorus directory was created and 20s audio chorus were added in the folder.

**Extracting Audio Features**
Using librosa library, extracted chorus features: song, chroma stft, chroma cqt, chroma cens, mfcc, rms, spectral centroid, spectral bandwidth, spectral contrast, spectral rolloff, tonnetz, zero crossing rate. A total of 897 rows and 519 columns.

# Data Pre-processing

At first the datasets containing both song detail and its features data was loaded. 
Removed .mp3 extension from the song name. After that, merged both the datasets and created a final dataframe.
Checked for the null values in the dataframe and removed the null value containing rows. 
Checked for dataframe info and its descriptive statics.
Calculated mean for each group of columns with same name by grouping them using loc function in pandas.
I had already dealt with duplicated columns so there was no need to check for duplicated values.
Exported the final data to a .csv format.

# Exploratory Data Analysis

**Univariate Analysis**

**Histogram plot**
Created histogram plot to check for the distribution of values in each column by selecting only numeric columns from the dataframe and found that only few column values were normally distributed. Most of the data were skewed.

**Box plot**
Created box plot to visually check for outlier values present in each data columns. Viewed the median, interquartile range and the range of the data. The central tendency and spread of the data was identified for numeric columns. 

**Density plot**
Selected only numeric columns and looped through each numeric column to plot density separately. Viewed smoothed estimate of the probability density function (PDF) of the data.

**Count plot**
Created count plot for categorical variables, for top categories. Filtered the dataframe for top 10 categories then created count plot.
For artist column: Taylor Swift had the most songs in the dataset followed by SZA, Bad Bunny, 21 Savage, The Weekend, J. Cole, Doja Cat, Luke Combs, Justin Bieber, Olivia Rodrigo.
For label column: The number of popular song was higher (700+) than unpopular song (around 200) by a large margin.

**Bivariate Analysis**

**Pie chart**
Crated a copy of the dataframe and replaced 1 with popular and 0 with unpopular in label. Calculated the number of popular and unpopular songs. Calculated total number of observations and then plotted pie chart. Viewed number distribution of popular and unpopular songs. 
Unpopular: 189 
Popular: 726
After viewing artists with highest number of songs in the dataframe by plotting bar graph, checked the popular and unpopular song percentage for the top artist (Taylor Swift).
Filtered through Taylor Swift’s songs and renamed the labels as popular and unpopular. Calculated the percentage distribution of the labels. Then, plotted pie chart.
Popular and Unpopular songs of Taylor Swift
Popular: 61.1%
Unpopular: 38.9%

**Pair plot**
Computed pairwise correlation for  'chroma_stft_mean', 'chroma_cqt_mean', 'chroma_cens_mean', 'mfcc_mean', 'rms_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_contrast_mean', 'spectral_rolloff_mean', 'tonnetz_mean', 'zero_crossing_rate_mean' and 'Label' from final_df. Viewed correlation matrix.
Created visualization to view relationship between all pairs of numeric variables using pair plot.
Spectral rolloff mean had positive linear relation with spectral centroid mean and spectral bandwidth mean.

**Multivariate Analysis**

**Heat map**
Visualized correlation matrix as heatmap. Each cell in the heatmap represented the correlation coefficient between two variables, with colors indicating the direction and strength of the correlation.
Value close to -1 represented strong negative relationship between variables
Value close to 0 represented no correlation between variables
Value close to 1 represented strong positive relationship between variables
Spectral rolloff mean and spectral bandwidth mean had the highest positive correlation which was 0.92
Mfcc mean and chroma stft mean had the highest negative correlation which was -0.53

**Principal Component analysis (PCA)**
Imported PCA and StandardScalar library for calculating PCA. Loaded the numerical columns from the data and assigned it as X, standardized the features, preformed principal component analysis adjusting the component to 2 dimension, visualized the PCA results then printed the explained variance ratio.
Explained variance ratio: [0.12323681 0.06299722]  
The total variance explained by the principal components was around 18.61% when we were getting 2 components. Generally, a higher percentage of explained variance is desirable as it indicates that the principal components capture a significant amount of information from the original data. We can increase the principal component accordingly to achieve variance close to 100%.

# Feature Engineering

It was a new notebook so imported required libraries then read the csv file. Checked info and descriptive statistics.
Encoded label as popular unpopular for computation of numerical columns
Values for Label (1, 0) to (Popular, Unpopular)
Dropped Unnamed and YouTube_URL columns. Checked for null values, shape and duplicated values.
Checked for outliers present in the dataframe using interquartile method. Calculated the number of outlier values in each column. Outlier was present in almost every column.
Selected numeric columns from the dataframe for calculation first quartile and third quartile, Set the threshold to 1.5, determined the outliers using IQR, divided the outlier column in columns with less than and greater than 100 outliers, replaced the outliers with median for the columns with less than 100 outliers and removed the outliers rows for the columns with more than 1oo outlier values.
After dealing with outlier checked again if any outlier value remained in the dataset.

**Scaling Data**
Scaled the data using MinMaxScalar. Applied min-max scalar to each numeric column in the dataframe excluding Label.
Data Scaling: Using MinMaxScaler we can scale our data in between 0 to 1

**Polynomial feature**
Imported and initialized polynomial features, extracted the numerical data from the dataframe, fitted and transformed the numerical data to polynomial features, generated column names for polynomial features, converted the result to dataframe and then concatenated the polynomial features dataframe with the original dataset (categorical columns).
We have acquired 140714 columns after preforming polynomial featuring.

**PCA**
Imported required libraries for PCA and imported numeric columns, extracted categorical columns and standardized the numerical features, initialized PCA with 50% variance explained, fitted PCA to scaled numerical data and transformed it, created dataframe for PCA results and then concatenated PCA dataframe with categorical columns and target variable (Label). 
Viewed final PCA dataframe.
Preformed encoding for Label (Popular, Unpopular) to (1, 0) before exporting the data to csv file.
Created a cumulative explained variance plot with 18 principal components and up to 0.50 cumulative explained variance ratio.
Preforming one-hot encoding for song and artist columns to use them during modeling.
Final shape: (262, 488)

# Model Training and Evaluation

It was a new notebook so imported required libraries then read the csv file. Checked info and descriptive statistics.

**Logistic Regression**
From sklearn library imported train_test_split, LogisticRegression, accuracy_score, classification_report and confusion_matrix.
Prepared X and y then split the data into training and testing sets with test size of 0.4 with random state at 42.
Trained the logistic model, fitted X_train and y_train then evaluated the model for its accuracy, classification report and confusion matrix.
Accuracy: 0.7904761904761904
Classification Report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        21
           1       0.80      0.99      0.88        84

    accuracy                           0.79       105
   macro avg       0.40      0.49      0.44       105
weighted avg       0.64      0.79      0.71       105

Confusion Matrix:
[[ 0 21]
 [ 1 83]]
Made prediction on train and test data and calculated their accuracies
Training Accuracy: 87%
Testing Accuracy: 79%
Model Accuracy:- The accuracy was approximately 79.04% meaning the model correctly predicted the target variable for about 79.04% of instances in the test set.
Model Evaluation:- For Unpopular: Precision: 0 (0% of the instances predicted as class 0 were actually class 0) Recall: 0 (0% of the actual class 0 instances were predicted as class 0) F1-score: 0 (harmonic mean of precision and recall) For Popular: Precision: 0.8 (80% of the instances predicted as class 1 were actually class 1) Recall: 0.99 (99% of the actual class 1 instances were predicted as class 1) F1-score: 0.88 (High F1-score indicates better performance in predicting class 1)

**Naïve Bayes**
From sklearn library imported GaussianNB for Naive Bayes.
Prepared X and y then split the data into training and testing sets with test size of 0.5 with random state at 42.
Trained the naive bayes model, fitted X_train and y_train then evaluated the model for its accuracy, classification report and confusion matrix.
Accuracy: 0.20610687022900764
Classification Report:
              precision    recall  f1-score   support

           0       0.17      0.70      0.27        27
           1       0.50      0.08      0.13       104

    accuracy                           0.21       131
   macro avg       0.33      0.39      0.20       131
weighted avg       0.43      0.21      0.16       131

Confusion Matrix:
[[19  8]
 [96  8]]
Made prediction on train and test data and calculated their accuracies
Training Accuracy: 100%
Testing Accuracy: 20%
The training accuracy of 1.0 means that our Naive Bayes model had perfectly learned the patterns in the training data. But, testing accuracy of 0.20 was quite low which suggests that the model was not performing well on unseen data. The low testing accuracy indicated that the model was not generalizing well and was not able to make accurate predictions on new data points.

**Decision Tree**
From sklearn library imported DecisionTreeClassifier.
Prepared X and y then split the data into training and testing sets with test size of 0.41 with random state at 42.
Trained the naive bayes model, fitted X_train and y_train then evaluated the model for its accuracy, classification report and confusion matrix.
Accuracy: 0.75
Classification Report:
              precision    recall  f1-score   support

           0       0.22      0.09      0.13        22
           1       0.80      0.92      0.85        86

    accuracy                           0.75       108
   macro avg       0.51      0.50      0.49       108
weighted avg       0.68      0.75      0.71       108

Confusion Matrix:
[[ 2 20]
 [ 7 79]]
Made prediction on train and test data and calculated their accuracies
Training Accuracy: 100%
Testing Accuracy: 75%
The training accuracy of 1.0 indicated that the decision tree model had perfectly fit the training data, correctly predicting all labels. However, with a testing accuracy of 0.75, the model seemed to generalize better to unseen data compared to the Naive Bayes model. This suggested that the decision tree model was more robust and was better for making accurate predictions on new data.

**Support Vector Machine (SVM)**
From sklearn library imported SVC for Support Vector Machine (SVM).
Prepared X and y then split the data into training and testing sets with test size of 0.31 with random state at 42.
Trained the naive bayes model, fitted X_train and y_train then evaluated the model for its accuracy, classification report and confusion matrix.
Accuracy: 0.7926829268292683
Classification Report:
              precision    recall  f1-score   support

           0       0.57      0.22      0.32        18
           1       0.81      0.95      0.88        64

    accuracy                           0.79        82
   macro avg       0.69      0.59      0.60        82
weighted avg       0.76      0.79      0.76        82

Confusion Matrix:
[[ 4 14]
 [ 3 61]]
Made prediction on train and test data and calculated their accuracies
Training Accuracy: 100%
Testing Accuracy: 79%
The training accuracy of 1.0 indicated that the SVM model had perfectly fit the training data, correctly predicting all labels. However, with a testing accuracy of 0.79, the model seemed to generalize better to unseen data compared to the Naive Bayes model and Decision Trees model. This suggested that the decision tree model was more robust and better for making accurate predictions on new data.

# Model Deployment
In progress...



