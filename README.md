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

* Explored the distribution of passer ratings
* Examined the relationships between features and the target variable (passer rating)

### Problem Formulation

* Input: Summary statistics of quarterback performance (e.g., completion percentage, yards per attempt, etc.)
* Output: Passer rating
* Models: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
* Loss, Optimizer, other Hyperparameters: Model-specific (e.g., number of estimators for Random Forest)




