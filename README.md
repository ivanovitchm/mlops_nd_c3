# Project Overview

In this project, we will apply the skills acquired in the ``Deploying a Scalable ML Pipeline in Production`` course to develop a classification model on publicly available [Census Bureau data](http://archive.ics.uci.edu/ml/datasets/Adult). 

We will create unit tests to monitor the model performance on various slices of the data. Then, we will deploy the final model using the FastAPI package and create API tests. Both the slice-validation and the API tests will be incorporated into a CI/CD framework using GitHub Actions. DVC will be used to manage the project dependencies (model, data, so on) and allow a reproducible pipeline.

## Environment Setup

Create a conda environment with ``environment.yml``:

```bash
conda env create --file environment.yml
```

To remove an environment in your terminal window run:

```bash
conda remove --name myenv --all
```

To list all available environments run:

```bash
conda env list
```

## Data Pipeline 

We will use DVC to manage and version the data processes that produce our final artifact. This mechanism allows you to organize the project better and reproduce your workflow/pipeline and results more quickly. The following steps are considered: a) ``data``, b) ``eda``, c) ``preprocess``, d) ``check data``, e) ``segregate``, f) ``train`` and g) ``evaluate``.

### Data

It is assumed the data has already been fetched and stored at ``pipeline/01_data``.

Before starting the next pipeline stage, EDA, clone (git clone), or create a new git repository (git init) to the project. Right after, run:

```bash
dvc init
```

and track and version the file ``census.csv`` using:

```bash
dvc add pipeline/01_data/census.csv
git add pipeline/01_data/.gitignore pipeline/01_data/census.csv.dvc
```

It is possible tracking data remotely with DVC. In this project we will use a ``S3 bucket`` as configuration. Some aditional steps are necessary to setup the CLI environment. 

1. Install the [AWS CLI tool](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html).
2. Sign in [AWS Amazon](https://aws.amazon.com/) using your user and password. 
3. From the Services drop down select ``S3`` and then click ``Create bucket``.
4. Give your bucket a name, the rest of the options can remain at their default.
5. To use your new ``S3 bucket`` from the AWS CLI you will need to create an ``IAM user`` with the appropriate permissions.
    - Sign in to the [IAM console](https://console.aws.amazon.com/iam/) or from the Services drop down on the upper navigation bar.
    - In the left navigation bar select ``Users``, then choose ``Add user``.
    - Give the user a name and select ``Programmatic access``.
    - In the permissions selector, search for S3 and give it ``AmazonS3FullAccess``.
    - ``Tags`` are optional and can be skipped.
    - After reviewing your choices, click ``create user``.
    - Configure your AWS CLI to use the Access key ID and Secret Access key.
    ```bash
    aws configure
    ```

Right after this setup, add the S3 remote bucket using:

```bash
dvc remote add name_of_remote_repo s3://name_of_s3_bucket
```

In our case, the configuration was:

```bash
dvc remote add s3remote s3://incomes3
```

To visualize the configuration run:

```bash
dvc remote list
```

### Exploratory Data Analysis (EDA)

For now, the artifacts generated by EDA are not tracked. EDA is only to understand the big-picture about the problem.  
