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

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

