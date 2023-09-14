# Personality Traits Analysis and Clustering


This project aims to Analyze personality trait data, use clustering analysis to identify different personality profiles within the dataset, wxplore a six-factor personality model, and compare the findings with the well-known Big Five personality model.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Clustering](#clustering)
- [Visualization](#visualization)
- [Correlation Analysis](#correlation-analysis)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [License](#license)

## Introduction

This project focuses on analyzing personality traits data and discovering patterns through clustering analysis. It involves preprocessing the data, performing K-means clustering, visualizing the clusters, and examining the correlation between different personality factors.

## Data

The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/code/akdagmelih/five-personality-clusters-k-means). The original dataset contains personality traits survey data. In this project, the survey responses are categorized into different traits such as extroversion, neuroticism, agreeableness, methodicalness, ambition, and openness.

## Data Preprocessing

Before the analysis, the raw data is preprocessed, which includes:
- Dropping irrelevant columns
- Handling missing values
- Dropping outliars (patiants which extremely not differentiate between different questions)
- Renaming columns
- Scaling the data using MinMaxScaler

## Clustering

The main analysis involves K-means clustering, where the scaled data is clustered into distinct groups based on the similarity of personality traits.

## Visualization

The project includes visualizations of:
- Bar plots and line plots to display the means of personality factors for each cluster
- <img width="1002" alt="image" src="https://github.com/shirashko/Neuropsychiatry/assets/130151997/7c553b0c-f997-4bcf-b0ea-4d9b9b00af85">

- 2D scatter plot using PCA to visualize the clusters
- <img width="582" alt="image" src="https://github.com/shirashko/Neuropsychiatry/assets/130151997/1869f10b-259e-444c-a780-1160e5a53298">


## Correlation Analysis

Correlation matrices are computed to examine relationships between different personality factors. Heatmaps are used to visualize the correlation values.
<img width="970" alt="image" src="https://github.com/shirashko/Neuropsychiatry/assets/130151997/e50b20be-0414-4d8f-931f-10c14d95dbd6">


## Getting Started

To run this project locally, follow these steps:

1. Install the required dependencies. 
2. Clone this repository.
3. Place the `data-final.csv` file in the same directory as the script from the resource we provided above.
4. Run the script 

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Original Code
The original code that this project is based on is included in the repository as "five-personality-clusters-k-means.ipynb". This code serves as the foundation for the clustering analysis in this project.

## Usage

1. Install the necessary dependencies using `pip install -r requirements.txt`.
2. Ensure the `data-final.csv` file is in the same directory.
3. Run the script using `python script_name.py`.

