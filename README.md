# Audionome
An AI-powered music genre classification system developed by [Simon Stijnen](https://github.com/SimonStnn) and [Lynn Delaere](https://github.com/LynnDelaere), for the course AI Machine Learning at VIVES University of Applied Sciences.

![DALLEImage](/app/static/DALLEImage.webp)

# Table of contents
- [Audionome](#audionome)
- [Table of contents](#table-of-contents)
- [Objectives](#objectives)
  - [Main objective](#main-objective)
  - [Sub-objectives](#sub-objectives)
- [Problem statement](#problem-statement)
- [Analysis](#analysis)
  - [Preliminary research](#preliminary-research)
  - [Data collection](#data-collection)
  - [Data preprocessing](#data-preprocessing)
  - [Model selection](#model-selection)
    - [Logistic regression](#logistic-regression)
    - [Stochastic gradient descent classifier](#stochastic-gradient-descent-classifier)
    - [Random forest classifier](#random-forest-classifier)
    - [Support vector classifier](#support-vector-classifier)
    - [K-nearest neighbors](#k-nearest-neighbors)
    - [Decision tree](#decision-tree)
    - [Gradient boosting](#gradient-boosting)
- [Results](#results)
  - [Model Evaluation](#model-evaluation)
  - [Model Secetion](#model-secetion)
- [Conclusion](#conclusion)
- [Future work](#future-work)
- [Tools](#tools)
  - [Technologies](#technologies)
  - [Libraries](#libraries)
  - [Hardware requirements](#hardware-requirements)
  - [Software requirements](#software-requirements)
  - [Deployment](#deployment)
- [References](#references)
  

# Objectives
## Main objective
Our project focuses on developing a machine learning model that can classify music based on its genre.

For this, we are going to use an existing dataset: the GTZAN dataset. This dataset consists of 10 different genres, each containing 100 audio samples, along with two csv files containing the features extracted from the audio files. By using an existing dataset, we can focus on analyzing its structure and understanding the types of features extracted. This knowledge will be valuable if we decide to build our own audio dataset in the future, as we will have a better understanding of how to create a well-structured dataset.

For the machine learning model itself, we will use the Scikit-Learn Python module. This is a simple and efficient library for data analysis, pre-processing data, and model selection. It is built on SciPy, NumPy, Joblib and Threadpoolctl. Scikit-Learn is also open-source and has a large community, which makes it easy to find help and examples online.

## Sub-objectives
- Analyze the GTZAN dataset and understand the structure of the data.
- Buil a user-friendly interface that allows users to upload an audio file and get the genre classification.
- Test the model with different audio files and evaluate its performance.
- Investigate the possibility of using deep learning models for music genre classification.

# Problem statement
Correctly classifying music by genre is not always straightforward. Different pieces of music often share overlapping characteristics and can be subjectively assessed. Our project aims to develop an understanding of the relevant features that can be extracted from music and how these features can be used to create a machine learning model that can classify music by genre.

Music genre classification presents several challenges that our project seeks to address. First, there's the issue of feature extraction - identifying which audio characteristics are most relevant for distinguishing between genres. Traditional approaches rely on manually engineered features such as spectral centroids, MFCCs, chroma features, and zero-crossing rates, but determining the optimal combination and representation of these features is complex.

Furthermore, genre boundaries are inherently subjective and often fuzzy. What one person considers rock, another might classify as pop or alternative. This subjectivity creates challenges in creating reliable ground truth for training models. Additionally, genres evolve over time, with new sub-genres emerging and existing ones blending, making classification a moving target.

For music streaming services, recommendation systems, and music libraries, accurate genre classification can significantly improve user experience through better organization and discoverability. Researchers in musicology can benefit from automated genre classification tools to analyze large collections. Content creators and producers can use such systems to properly tag and categorize their music for distribution platforms.

Our target audience includes:
- Music streaming services seeking to improve their recommendation systems
- Music researchers and analysts who need to process large music libraries
- Individual users who want to organize their personal music collections
- Content creators looking to properly categorize their music for distribution
- Music educators who can use the tool for teaching about genre characteristics

# Analysis
## Preliminary research
To establish a solid foundation for our project, we researched existing projects that classify music by genre. We found that music genre classification is a well-researched area in machine learning. Several studies have used audio features to classify music genres with high accuracy. We also found that the GTZAN dataset is a popular dataset for music genre classification. It contains 1000 audio tracks, each labeled with a genre.
Several projects we have found and based on them we will develop our project:
- [Work w/ Audio Data: Visualise, Classify, Recommend](https://www.kaggle.com/code/andradaolteanu/work-w-audio-data-visualise-classify-recommend)
- [Music Genre Classification using Machine Learning](https://www.geeksforgeeks.org/music-genre-classifier-using-machine-learning/)
- [Let's tune🎧the music🎵🎶with CNN🎼XGBoost🎷🎸🎻🎺](https://www.kaggle.com/code/aishwarya2210/let-s-tune-the-music-with-cnn-xgboost)
- [Music Genre Classification: Training an AI model](https://arxiv.org/html/2405.15096v1)
- [Music Genre Classification](https://www.kaggle.com/code/jvedarutvija/music-genre-classification)

## Data collection
The GTZAN dataset is a well-known dataset in the field of music genre classification. It contains 1000 audio tracks, each 30 seconds long, labeled with one of 10 genres. The genres included in the dataset are: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. The dataset also includes two CSV files containing the features extracted from the audio files. These features include MFCCs, chroma features, spectral contrast, and tonnetz features.
You can download the GTZAN dataset from the following link: [GTZAN dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

## Data preprocessing
Due to the fact that we are using an existing dataset we don't need to extract the features from the audio files anymore. The GTZAN dataset already contains the features extracted from the audio files. But we still wanted to understand the structure of the data and how the features are represented. Some of the extracted features include:
- MFCCs (Mel-frequency cepstral coefficients): These are a type of feature widely used in speech and audio processing. They represent the short-term power spectrum of a sound.
- Chroma features: These are used to represent the energy distribution of the chroma scale, which is a 12-tone scale that represents all the notes in an octave.
- Spectral features (centroid, bandwidth, centroid, rolloff): These features describe the shape of the sound spectrum.

We made a correlation matrix to see how the features are related to each other. This helped us understand which features are most important for classifying music by genre. 

![CorrelationMatrix](/app/static/correlationMatrix.png)

Key observations:
- The diagonal of the correlation matrix shows that each feature is perfectly correlated with itself.
- The correlation matrix is perfectly symmetrical, as expected.
- The first group of features (chroma_stft, rmse, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate) are positively correlated with each other.
- The second group of features (mfcc1 to mfcc20) show an interesting pattern where the adjacent features are highly correlated with each other.
- There is a noticeable low correlation between the mfcc2 feature and the first group of features.

To visualize the data in a lower-dimensional space, we will use PCA to reduce the dimensionality of the features while preserving as much information as possible. PCA transforms the data into a new coordinate system where the main patterns become clearer and variables are less correlated.

For standardization, we will use the StandardScaler class from Scikit-learn. This will transform the data so that it has a mean of 0 and a standard deviation of 1. For each feature in the dataset, the StandardScaler performs the following operation:
- Calculate the mean of all values in the feature.
- Subtract the mean from each value in the feature.
- Calculate the standard deviation of all values in the feature.
- Divide each value in the feature by the standard deviation.
The result is a dataset where each feature has a mean of 0 and a standard deviation of 1.

This standardization is also important for training machine learning models, as it ensures that all features have the same scale. If the features are not standardized, features with larger values may dominate the learning process.

After standardization, we will apply PCA to the data. We will use the PCA class from Scikit-learn to perform PCA. The PCA class works as follows:
- The mean of each feature is subtracted from the data.
- The covariance matrix of the data is calculated.
- The eigenvectors and eigenvalues of the covariance matrix are computed. This is done using the singular value decomposition (SVD) method.
- The eigenvectors are sorted by their corresponding eigenvalues in descending order.
- The data is projected onto the principal components.

![PCA](/app/static/PCAPlot.png)

The PCA plot shows that most of the data is clustered together, some genres are more spread out than others. This suggests that these genres have more variability in their features. In the middle of the plot, more genres are clustered together, indicating that they share similar features.

Before we can train our machine learning model, we need to split the data into training, validation and testing sets. We will use the train_test_split function from Scikit-learn to split the data. This function randomly shuffles the data and splits it into training, validation and testing sets. We will use 80% of the data for training, 10% for validation and 10% for testing.

## Model selection
Before we started building our model, we researched different machine learning algorithms that could be used for classifying music by genre. We considered the following algorithms:
- Logistic regression
- Stochastic gradient descent
- Random forest
- Support vector machine
- K-nearest neighbors
- Decision tree
- Gradient boosting

We chose these algorithms because they are commonly used for classification tasks and are supported by the Scikit-Learn library. They are also relatively easy to implement and tune. We planned to experiment with different algorithms to see which one performed best on the GTZAN dataset.

### Logistic regression
Logistic regression is a supervised machine learning algorithm used for classification tasks. It helps predict the probability that a data point belongs to a specific class. It’s commonly used for binary classification problems—where the target variable has two possible outcomes, like yes/no or 0/1. However, it can also be used for multi-class classification problems. 
The logistic regression model that is used in the Scikit-Learn library has several hyperparameters that can be tuned to improve performance. Some of the key hyperparameters include:
- penalty: This hyperparameter specifies the norm used in the penalization term. It can be set to 'l1' or 'l2'.
- C: This hyperparameter specifies the inverse of the regularization strength. A smaller value of C indicates stronger regularization.
- solver: This hyperparameter specifies the algorithm used for optimization. It can
be set to 'newton-cg', 'lbfgs', 'liblinear', 'sag', or 'saga'.

### Stochastic gradient descent classifier
The stochastic gradient descent (SGD) classifier is a linear classifier that uses stochastic gradient descent to optimize the loss function. It is suitable for large-scale machine learning problems and can be used for both binary and multi-class classification tasks. The SGD classifier has several hyperparameters that can be tuned to improve performance. Some of the key hyperparameters include:
- loss: This hyperparameter specifies the loss function to be optimized. It can be set to 'hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', or 'squared_loss'.
- penalty: This hyperparameter specifies the norm used in the penalization term. It can be set to 'l1', 'l2', or 'elasticnet'.
- learning_rate: This hyperparameter specifies the learning rate schedule. It can be set to 'constant', 'optimal', 'invscaling', or 'adaptive'.

### Random forest classifier
The random forest classifier is an ensemble learning method that combines multiple decision trees to improve performance. It is suitable for both binary and multi-class classification tasks and can handle large datasets with high dimensionality. The random forest classifier has several hyperparameters that can be tuned to improve performance. Some of the key hyperparameters include:
- n_estimators: This hyperparameter specifies the number of trees in the forest. A larger number of trees can improve performance but may increase training time.
- max_depth: This hyperparameter specifies the maximum depth of the trees in the forest. A larger maximum depth can lead to overfitting.
- min_samples_split: This hyperparameter specifies the minimum number of samples required to split an internal node. A smaller value can lead to overfitting.

### Support vector classifier
The support vector classifier (SVC) is a supervised machine learning algorithm used for classification tasks. It is suitable for both binary and multi-class classification problems and can handle large datasets with high dimensionality. The SVC algorithm has several hyperparameters that can be tuned to improve performance. Some of the key hyperparameters include:
- C: This hyperparameter specifies the regularization parameter. A smaller value of C indicates stronger regularization.
- kernel: This hyperparameter specifies the kernel function used for mapping the input data into a higher-dimensional space. It can be set to 'linear', 'poly', 'rbf', 'sigmoid', or 'precomputed'.
- gamma: This hyperparameter specifies the kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. A smaller value of gamma indicates a larger influence of the training samples.

### K-nearest neighbors
The k-nearest neighbors (KNN) algorithm is a simple and intuitive machine learning algorithm used for classification tasks. It classifies data points based on the majority class of their k nearest neighbors. The KNN algorithm has one hyperparameter that can be tuned to improve performance:
- n_neighbors: This hyperparameter specifies the number of neighbors used for classification. A larger number of neighbors can improve performance but may increase computational cost.
- weights: This hyperparameter specifies the weight function used for prediction. It can be set to 'uniform' or 'distance'.

### Decision tree
The decision tree algorithm is a supervised machine learning algorithm used for classification tasks. It builds a tree-like structure to represent the decision process and classify data points. The decision tree algorithm has several hyperparameters that can be tuned to improve performance. Some of the key hyperparameters include:
- criterion: This hyperparameter specifies the criterion used for splitting nodes. It can be set to 'gini' or 'entropy'.
- max_depth: This hyperparameter specifies the maximum depth of the tree. A larger maximum depth can lead to overfitting.
- min_samples_split: This hyperparameter specifies the minimum number of samples required to split an internal node. A smaller value can lead to overfitting.

### Gradient boosting
The gradient boosting algorithm is an ensemble learning method that combines multiple weak learners to improve performance. It is suitable for both regression and classification tasks and can handle large datasets with high dimensionality. The gradient boosting algorithm has several hyperparameters that can be tuned to improve performance. Some of the key hyperparameters include:
- n_estimators: This hyperparameter specifies the number of boosting stages. A larger number of boosting stages can improve performance but may increase training time.
- learning_rate: This hyperparameter specifies the learning rate used for updating the weights of the weak learners. A smaller learning rate can improve generalization.

# Results
## Model Evaluation
We evaluated the performance of each model using the following metrics:
- Accuracy: This metric measures the proportion of correctly classified instances.
- Confusion matrix: This matrix shows the number of true positive, true negative, false positive, and false negative predictions.

The results of the model evaluation are as follows:
- Logistic regression: The logistic regression model achieved an accuracy of 72.17% on the test set, with the parameters max_iter = 200, solver = 'lbfgs' and the penalty set to 'l2'. The confusion matrix showed that the model struggled with some genres like country, disco and rock. Where the true positives where far less then the rest of the genres.

![LogisticRegression](/app/static/LogisticRegression.png)

- Stochastic gradient descent classifier: The stochastic gradient descent classifier achieved an accuracy of 67.97% on the test set, with the parameters max_iter = 200, a constant learning rate of 0.01, and the penalty set to 'l2'. The confusion matrix showed that the model struggles on the same genres as the logistic regression model, where the true positives of theses genres where even less then the logistic regression model.

![StochasticGradientDescent](/app/static/StochasticGradientDescent.png)

- Random forest classifier: The random forest classifier achieved an accuracy of 78.48% on the test set, with the parameters n_estimators = 81 and max_depth = 10. The confusion matrix showed that the model performed better then the previous models, but still struggled with the genres country and disco. But in general this model preformed better then the previous models.

![RandomForest](/app/static/RandomForest.png)

- Support vector classifier: The support vector classifier achieved an accuracy of 91.99% on the test set, with the parameters C = 10, kernel = 'rbf', and gamma = 'scale'. The confusion matrix showed that the model performed well for most genres, but struggled a bit with the genre country. But so far this model preformed the best of all the models.

![SupportVectorClassifier](/app/static/SupportVectorClassifier.png)

- K-nearest neighbors: The k-nearest neighbors algorithm achieved an accuracy of 91.09% on the test set, with the parameter n_neighbors = 6. The confusion matrix showed that the model performed well for most genres, but struggled with the genre country. But this model preformed almost as good as the support vector classifier.

![KNearestNeighbors](/app/static/KNearestNeighbors.png)

- Decision tree: The decision tree algorithm achieved an accuracy of 66.17% on the test set, with the parameters max_depth = 15, criteriion = 'enthorpy', and the class_weight set to 'balanced'. The confusion matrix showed that the model struggled a lot with the genres country, disco and rock. This model preformes so far the worst of all the models.

![DecisionTree](/app/static/DecisionTree.png)

- Gradient boosting: The gradient boosting algorithm achieved an accuracy of 87.69% on the test set, with the parameters n_estimators = 300, learning_rate = 0.2, and a loss function set to 'log_loss'. The confusion matrix showed that the model performed well for most genres, but struggled with the genre country and rock. This model preformed better then the decision tree model, but not as good as the support vector classifier and the k-nearest neighbors algorithm.

![GradientBoosting](/app/static/GradientBoosting.png)

We've plotted each models accuracy on the training, testing and validation set. The plot shows that the support vector classifier and the k-nearest neighbors algorithm performed the best on the test set, with an accuracy of 91.99% and 91.09% respectively. 
In the plot we can clearly see that most models tend to overfit on the training set, this you can see by the high accuracy on the training set and the lower accuracy on the test set. Only with the logistic regression model and the stochastic gradient descent classifier this is not the case.
We can also see that the decision tree model performed the worst on the test set, with an accuracy of 66.17%. The plot gives us a good overview of how each model performed on the different sets.

In the second plot, we've plotted the metrics comparison of each model. The plot shows that the support vector classifier and the k-nearest neighbors algorithm performed the best on the test set, with an accuracy of 91.99% and 91.09% respectively. The plot also shows the differens between the training accuracy and the test accuracy, this shows that most models tend to overfit on the training set. Where the smallest gap between the training accuracy and the test accuracy is with the logistic regression model and the stochastic gradient descent classifier.

![ModelComparison](/app/static/ModelComparison.png)

![ModelComparisonMetrics](/app/static/ModelComparisonMetrics.png)

## Model Secetion
After evaluating the performance of each model, we selected the support vector classifier as the best model. The support vector classifier achieved an accuracy of 91.99% on the test set, which was the highest accuracy of all the models we tested. The gap between the training accuracy and the test accuracy was also relatively small, indicating that the model did not overfit on the training set. The confusion matrix showed that the model performed well for most genres, overall the support vector classifier was the best model we tested for classifying music by genre.

# Conclusion
We can conclude that we have succesfully developed a machine learning model that is capable of accurately classifying music by genre using audio features. Reflecting on our initial objectives, we can conclude that we have largely achieved our main goal of creating a functional genre classification system with good accuracy.

Our model achieved an accuracy of 91.99% on the test set using the support vector classifier. Which is a respectable performance given the complexity of the task and the known limitations of the GTZAN dataset. This level of accuracy demonstrates the potential of machine learning models for music genre classification and suggests that further improvements could be made with more advanced feature extraction techniques and larger datasets.

Our sub-objectives are also largely fulfilled. We have analyzed the GTZAN dataset and gained a better understanding of the features extracted from audio files. We have built a user-friendly interface that allows users to upload an audio file and get the genre classification. We have tested the model with different audio files and evaluated its performance. We have also investigated the possibility of using deep learning models for music genre classification, although we did not implement them in this project.

However, we also encountered some challenges during the project. The GTZAN dataset is relatively small and may not be representative of all music genres. This could limit the generalizability of our model to other datasets. We also found that some genres were more difficult to classify than others, which suggests that further research is needed to improve the performance of the model on these genres.

Overall, we are satisfied with the results of our project and believe that it has the potential to be further developed and improved in the future. We have gained valuable experience in working with audio data and machine learning models, and we look forward to applying this knowledge to future projects.

# Tools
## Technologies
In our project, we used the following technologies:
- Python: We will use Python as the main programming language for our project. 
- Jupyter Notebook: We will use Jupyter Notebook for data analysis, visualization, and model training.

## Libraries
We used the following libraries in our project:
- Scikit-Learn: We used Scikit-Learn for building and training our machine learning model.
- Pandas: We used Pandas for data manipulation and preprocessing.
- NumPy: We used NumPy for numerical operations and array manipulation.
- Matplotlib: We used Matplotlib for data visualization.
- Seaborn: We used Seaborn for statistical data visualization.
- Librosa: We used Librosa for audio analysis and feature extraction.
- IPython: We used IPython for interactive computing.
- Pickle: We used Pickle for saving and loading our trained model.
- Streamlit: We used Streamlit to create a user-friendly interface for our model.

## Hardware requirements
Our project can be run on a standard laptop or desktop computer. We did not require any specialized hardware for training our model. 

## Software requirements
To run our project, you will need to have Python version 3.12.9 installed on your computer, with the following libraries:
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Librosa
- IPython
- Pickle
- Streamlit

You can install these libraries using the following command:
```
pip install -r requirements.txt requirements.training.txt
```
## Deployment
We plan to deploy our music genre classifiation system as a web application using Streamlit. This will allow users to upload an audio file our choose an audio file from a list of pre-selected audio files and get the genre classification. We will also provide a page with the reasearch and analysis of the project, as well as the results and conclusions.

To ensure consitency and reproducibility, we will use Docker to containerize our application. To build the Docker image, you can use the following command:
```
docker compose up -d
```
The application will be accessible at http://localhost:8501.

# References
- [GTZAN dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)
- [GeeksforGeeks](https://www.geeksforgeeks.org/music-genre-classifier-using-machine-learning/)
- [Kaggle feature extration](https://www.kaggle.com/code/ashishpatel26/feature-extraction-from-audio)
- [Scikit-Learn documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas documentation](https://pandas.pydata.org/docs/)
- [NumPy documentation](https://numpy.org/doc/)
- [Matplotlib documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn documentation](https://seaborn.pydata.org/)
- [Librosa documentation](https://librosa.org/doc/main/index.html)
- [IPython documentation](https://ipython.org/documentation.html)
- [Pickle documentation](https://docs.python.org/3/library/pickle.html)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Docker documentation](https://docs.docker.com/)
