# Codon Kingdom Classifier: Predicting Species Kingdoms from DNA Codon Frequencies

This project aims to classify species into different kingdoms based on the frequencies of codons in their DNA sequences. The dataset contains codon usage frequencies for various species, and the goal is to build a classification model to predict the kingdom to which a species belongs.

## Impact 

The model can help identify specific codon usage patterns that are characteristic of certain kingdoms or types of organisms. This can be instrumental in genetic research, facilitating the discovery of genetic markers that can be used for species identification. 

The application of various machine learning models (such as Random Forest, KNN, and Gaussian Naive Bayes) to classify species based on codon frequencies contributes to the development and refinement of computational methods in bioinformatics. These methods can be applied to other datasets and biological questions, enhancing the toolkit available to researchers.

## Dataset

The dataset is a CSV file with 13,026 entries and 69 columns. Each row represents a species, and the columns include:

- `Kingdom`: is a 3-letter code corresponding to `xxx' in the CUTG database name: 'arc'(archaea), 'bct'(bacteria), 'phg'(bacteriophage), 'plm' (plasmid), 'pln' (plant), 'inv' (invertebrate), 'vrt' (vertebrate), 'mam' (mammal), 'rod' (rodent), 'pri' (primate), and 'vrl'(virus).

- `DNAtype`: denoted as an integer for the genomic composition in the species: 0-genomic, 1-mitochondrial, 2-chloroplast, 3-cyanelle, 4-plastid, 5-nucleomorph, 6-secondary_endosymbiont, 7-chromoplast, 8-leucoplast, 9-NA, 10-proplastid, 11-apicoplast, and 12-kinetoplast.

- `SpeciesID`: Numerical identifier for each species.

- `Ncodons`: Codon frequencies are normalized to the total codon count, hence the number of occurrences divided by 'Ncodons' is the codon frequencies listed in the data file. 

- `SpeciesName`: Name of the species.

- `UUU` to `UGA`: codon frequencies recorded as floats (with decimals in 5 digits).


Hallee, L., Khomtchouk, B.B. Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life. Sci Rep 13, 2088 (2023). https://doi.org/10.1038/s41598-023-28965-7.
https://archive.ics.uci.edu/dataset/577/codon+usage


## Project Structure

- `data/`: Directory containing the dataset.
  - `codon_usage.csv`: Raw dataset file.
  - `codon_df_clean.csv`: Clean dataset for EDA and ML models.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, and modeling.
  - `01-data-loading-cleaning.ipynb`:Data cleaning and preprocessing.
  - `02-eda.ipynb`: Initial data exploration and visualization.
  - `03-modelling`: Machine Learning models.
- `model`: ML models as a pkl.
- `App`: Streamlit documents. 
- `docs/figures`:Relevant figures from the notebooks.
- `references/paper`: Paper on which the study was base on.
- `README.md`: Project overview and instructions.

## Machine Learning Models
We have built and evaluated several machine learning models to classify species based on codon frequencies:

- **Gaussian Naive Bayes**: A probabilistic classifier based on applying Bayes' theorem with the assumption of independence between every pair of features.For this case it will be our baseline model. 

- **Logistic Regression**: A linear model for binary classification that can be extended to multiclass classification problems. For this case, 5 clasess. 

- **K-Nearest Neighbors (KNN)**: A non-parametric method used for classification by comparing the distance of a point to the points in the training set.

- **Random Forest**: An ensemble method that operates by constructing multiple decision trees and outputs the mode of the classes.

- **Clustering**: Unsupervised learning techniques to group species based on codon frequencies, providing insights into natural groupings in the data.

-**XGBoost**: An ensemble learning method that combines multiple weak learners (decision trees by default) to create a strong classifier. It iteratively builds new models that complement the errors of previous models, focusing on instances where previous models have performed poorly.

-**Ensemble Model** (XGBoost, Logistic Regression and SVC): Ensemble learning combines predictions from multiple machine learning models to produce better results than any single model. In this case, we're using a hard voting classifier that combines predictions based on the majority class label predicted by each individual classifier (XGBoost, Logistic Regression, and Support Vector Classifier).

| Model                 | Parameters                                                                                                               | Score | Precision (Average) | Recall (Average) |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------|-------|---------------------|------------------|
| GaussianNB            | StandarScaler                                                                                                            | 65%   | 62%                 | 64%              |
| Logisti Regression    | StandarScaler, PCA = 63, C = 1, cv =5                                                                                    | 90%   | 89%                 | 86%              |
| KNN                   | StandarScaler, n_neighbors = 3, metric = manhattan, weights = distance, cv = 5                                           | 96%   | 95%                 | 94%              |
| Random Forest(PCA)    | StandarScaler, n_components = 64, max_depth = 35, min_sample_leaf = 1, min_sample_split = 2, n_estimators = 350,  cv = 5 | 94%   | 95%                 | 85%              |
| Random Forest(No PCA) | StandarScaler, max_depth = 35,  min_sample_leaf = 1, min_sample_split = 2, n_estimators = 350,  cv = 5                   | 74%   | 82%                 | 61%              |
| XGBoost               |                                                                                                                          | 95%   | 95%                 | 88%              |
| Ensamble Model        | XGBoost, Logistic Regression, SVC, voting = hard                                                                         | 95%   | 96%                 | 89%              |

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

