# Project Overview 

In this project, we will apply the skills acquired in the ``Deploying a Scalable ML Pipeline in Production`` course to develop a classification model on publicly available [Census Bureau data](http://archive.ics.uci.edu/ml/datasets/Adult). 

We will create unit tests to monitor the model performance on various slices of the data. Then, we will deploy the final model using the FastAPI package and create API tests. Both the slice-validation and the API tests will be incorporated into a CI/CD framework using GitHub Actions. All source code, dataset and the model will be versioned using git and DVC.

## Environment Setup

Create a conda environment with ``environment.yml``:

```bash
conda env create --file environment.yml
```

To remove an environment in your terminal window run:

```bash
conda remove --name myenv --all
```

To list all environment run:

```bash
conda env list
```
