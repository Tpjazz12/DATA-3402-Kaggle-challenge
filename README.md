# NFL Quarterback Performance Prediction

This repository contains an analysis of NFL quarterback performance and uses machine learning models to predict passer ratings.

## Overview

The goal of this project is to predict the passer rating of NFL quarterbacks based on their historical performance data. Three different machine learning models are compared: Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor. The best model, based on the R-squared score and Mean Squared Error, is the Random Forest Regressor with an R-squared score of 0.95 and a Mean Squared Error of 90.96.

## Summary of Workdone

### Data:

 * Type: Tabular data containing NFL quarterback statistics
 * Input: CSV file of features, output: passer rating
 * Size: 8525 data points
 * Instances (Train, Test, Validation Split): 80% for training, 20% for testing

### Preprocessing / Clean up:

* Removed unnecessary columns
* Replaced missing values with the mean value of the column
* Grouped data by player and calculated summary statistics

### Data Visualization
* Create new Feature Engineering that potentially improve the models
![image](https://user-images.githubusercontent.com/111667888/236652066-b0eec2eb-7005-450c-811c-5962dc7b2658.png)
* Explored the distribution of passer ratings
* Examined the relationships between features and the target variable (passer rating)
![image](https://user-images.githubusercontent.com/111667888/236652074-71d12918-2e2d-4234-b710-4c33f94f2c85.png)

### Problem Formulation

* Input: Summary statistics of quarterback performance (e.g., completion percentage, yards per attempt, etc.)
* Output: Passer rating
* Models: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
* Loss, Optimizer, other Hyperparameters: Model-specific (e.g., number of estimators for Random Forest)

### Training

* Trained models using Python and scikit-learn
* Trained/tested models using an 80%/20% train/test split
* Compared models based on Mean Squared Error and R-squared score

### Performance Comparison

| Model                    | Mean Squared Error | Root Mean Squared Error | R-squared |
|--------------------------|--------------------|-------------------------|-----------|
| Linear Regression        | 167.76             | 12.95                   | 0.90      |
| Random Forest Regressor  | 90.96              | 9.54                    | 0.95      |
| Gradient Boosting        | 91.75              | 9.58                    | 0.95      |

### Conclusions

Both the Gradient Boosting Regressor and Random Forest Regressor models provide strong predictions of NFL quarterback passer ratings, with Random Forest Regressor having a marginally lower error rate. However, the choice between these two models could depend on additional factors, such as the trade-off between model complexity and interpretability, or the time it takes to train and make predictions.

### Future Work

* Experiment with additional features, such as player-specific and team-specific variables
* Explore other machine learning models and techniques, such as neural networks or feature selection methods
* Investigate the impact of different hyperparameter settings on model performance

## How to reproduce results

1. Clone the repository
2. Install required Python packages
3. Download and preprocess the data
4. Train the models and evaluate their performance
5. Compare the results and draw conclusions

### Overview of files in repository

* EDA.ipynb: Explore Dataset contains "Quarterback Passer Rating Over the Years" graph also "Correlation Heatmap for QB Performance Metrics", Explain about the feature engineer I choose for trainning and testing.
![image](https://user-images.githubusercontent.com/111667888/236651929-3c1c8e43-5874-4029-856b-016be6e037fb.png)


* Gradiant-Boosting-Regressor.ipynb: Trains and evaluates the Gradient Boosting Regressor model
* Linear-Regression-QB-Performance.ipynb: Trains and evaluates the Linear Regression model
* Random-Forest-Regressor-QB-Performance.ipynb: Trains and evaluates the Random Forest Regressor model

### Software Setup

* Python 3.7 or later
* Install required packages: pandas, numpy, matplotlib, seaborn, scikit-learn

### Data

Download the NFL quarterback performance data from Kaggle in NFL Statistics file

### Training

* Follow the instructions in the respective notebooks (Gradiant-Boosting-Regressor.ipynb, Linear-Regression-QB-Performance.ipynb, Random-Forest-Regressor-QB-Performance.ipynb) to train the models
* Adjust the hyperparameters as needed to optimize the model performance

### Performance Evaluation

Run all the notebook to compare the performance of the different models
Analyze the results and draw conclusions based on the performance metrics (Mean Squared Error, Root Mean Squared Error, R-squared)

## Citations
* NFL Statistics: https://www.nfl.com/
* Kaggle challenge: https://www.kaggle.com/datasets/kendallgillies/nflstatistics?select=Career_Stats_Passing.csv



