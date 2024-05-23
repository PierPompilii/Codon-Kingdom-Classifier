# Usage

This is a template repository for quickly setting up a data science project
It includes a simple folder structure and a conda environment for isolated dependency management.

1. Start a new repo using this template
2. Update your `LICENSE` file.
3. Update your `README.md` file. 
4. Set up and activate conda environment
    1. Rename your conda environment in the `./conda.yml` file.
    2. Add/change any dependencies and their versions in the `./conda.yml` file.
    3. Set up your conda environment and activate it by running:
        ```bash
        conda env create -f conda.yml
        conda activate <your-env-name>
        ```
5. Add your own scripts in `./src/`
6. Add your own notebooks in `./notebooks/`
7. Add your own data in `./data/`
    gitignore will ignore the data folder when you push to github
    save a copy of your raw and process data, 
    pickled models in a Google Drive folder
    and add the link in the gdrive-links.md file
8. Add your project documents, reports, presentation pdfs in the `./docs`
9. Add your references (tutorials, papers, books etc.) in `./references` 

------------------------------------------------------------------------------

## Project Title
=========================

### Executive Summary

... Define the problem
... What is the data science opportunity
... Key takeaways

### Demo

... Show your work:
...     Data visualisations
...     Interactive demo (e.g., stremlit app)
...     Short video of users trying out the solution


### Methodology

... High-level diagrams of entire process:
...     various data processing steps
...     various modelling directions
...     various prototyping directions


### Organization

#### Repository 

* `data` 
    - contains link to copy of the dataset (stored in a publicly accessible Google Drive folder)
    - saved copy of aggregated / processed data as long as those are not too large (> 10 MB)

* `model`
    - joblib dump of final model / model object

* `notebooks`
    - contains all final notebooks involved in the project

* `docs`
    - contains final report which summarises the project

* `references`
    - contains papers / tutorials used in the project

* `src`
    - Contains the project source code (refactored from the notebooks)

* `.gitignore`
    - Part of Git, includes files and folders to be ignored by Git version control

* `conda.yml`
    - Conda environment specification

* `Makefile`
    - Automation script for the project

* `README.md`
    - Project landing page (this page)

* `LICENSE`
    - Project license

#### Dataset

... Google Drive links to datasets and pickeled models

### Credits & References

... Include any personal learning

------------------------------------------------------------------------------