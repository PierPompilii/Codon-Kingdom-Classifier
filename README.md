# Codon Frequency Classification Project

This project aims to classify species into different kingdoms based on the frequencies of codons in their DNA sequences. The dataset contains codon usage frequencies for various species, and the goal is to build a classification model to predict the kingdom to which a species belongs.

## Dataset

The dataset is a CSV file with 13,026 entries and 69 columns. Each row represents a species, and the columns include:

- `Kingdom`: Biological classification (e.g., Animalia, Plantae).
- `DNAtype`: Type of DNA (e.g., mitochondrial, nuclear).
- `SpeciesID`: Numerical identifier for each species.
- `Ncodons`: Total number of codons in the species' genome.
- `SpeciesName`: Name of the species.
- `UUU` to `UGA`: Frequencies of each codon in the species' genome.


Hallee, L., Khomtchouk, B.B. Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life. Sci Rep 13, 2088 (2023). https://doi.org/10.1038/s41598-023-28965-7.
https://archive.ics.uci.edu/dataset/577/codon+usage


## Project Structure

- `data/`: Directory containing the dataset.
  - `codon_frequencies.csv`: The dataset file.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, and modeling.
  - `exploration.ipynb`: Initial data exploration and visualization.
  - `preprocessing.ipynb`: Data cleaning and preprocessing.
  - `classification.ipynb`: Building and evaluating classification models.
- `src/`: Python scripts for data processing and modeling.
  - `data_processing.py`: Functions for loading and preprocessing data.
  - `modeling.py`: Functions for training and evaluating models.
- `README.md`: Project overview and instructions.

## Machine Learning Models
We have built and evaluated several machine learning models to classify species based on codon frequencies:

- Gaussian Naive Bayes: A probabilistic classifier based on applying Bayes' theorem with the assumption of independence between every pair of features.

- Logistic Regression: A linear model for binary classification that can be extended to multiclass classification problems.

- K-Nearest Neighbors (KNN): A non-parametric method used for classification by comparing the distance of a point to the points in the training set.

- Random Forest: An ensemble method that operates by constructing multiple decision trees and outputs the mode of the classes.

- Clustering: Unsupervised learning techniques to group species based on codon frequencies, providing insights into natural groupings in the data.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

