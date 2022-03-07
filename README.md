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

<center><img width="600" src="https://drive.google.com/uc?export=view&id=1a-nyAPNPiVh-Xb2Pu2t2p-BhSvHJS0pO"></center>

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

Right after this setup, add the ``S3 remote bucket`` using:

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

Them push tracking files using:

```bash
dvc push --remote s3remote
```

### Exploratory Data Analysis (EDA)

For now, the artifacts generated by EDA are not tracked. EDA is only to understand the big-picture about the problem. 

### Preprocess

To create pipelines, use ``dvc run`` to create stages. In each stage you can define dependencies, parameters, inputs, outputs, and specify the command that is run. In order to create the preprocess stage run:

```bash
dvc run -n preprocess \
        -d pipeline/03_preprocess/run.py \
        -d pipeline/01_data/census.csv \
        -o pipeline/01_data/preprocessing_data.csv \
        python pipeline/03_preprocess/run.py --input_artifact_name pipeline/01_data/census.csv \
                                             --output_artifact_name pipeline/01_data/preprocessing_data.csv
```

To track the changes with git, run:

```bash
git add dvc.yaml pipeline/01_data/.gitignore dvc.lock
```

If everything is successful, new DVC files and a new artifact are generated in the repository. ``dvc.lock`` and ``dvc.yaml`` are  used to manage the pipeline whereas the clean data, ``preprocessing_data.csv``, must be placed at ``pipeline/01_data``.

Now, given the data and pipeline are up to date is time to update the remote repository, please run:

```bash
dvc push --remote s3remote    
```

### Check Data

In this stage of the pipeline we will apply ``deterministic`` and ``non-deterministic`` tests to the ``preprocessing_data.csv``.

For most of the ``non-deterministic`` tests you need a reference dataset, and a dataset to be tested. This is useful when retraining, to make sure that the new training dataset has a similar distribution to the original dataset and therefore the method that was used originally is expected to work well.

We will use the [Kolmogorov-Smirnov](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) test for goodness of fit. Remember that the 2 sample KS test is used to test whether two vectors come from the same distribution (null hypothesis), or from two different distributions (alternative hypothesis), and it is non-parametric.


All tests will be implemented using the ``pytest``. An important aspect when using ``pytest`` is understanding the ``fixture's scope`` works.

The scope of the ``fixture`` can have a few legal values, described [here](https://docs.pytest.org/en/6.2.x/fixture.html#fixture-scopes). We are going to consider only ``session`` but it possible to use ``function``. In the former, the fixture is executed only once in a pytest session and the value it returns is used for all the tests that need it; with the latter, every test function gets a fresh copy of the data. This is useful if the tests modify the input in a way that make the other tests fail, for example. Let's see this more closely run this DVC pipeline:

```bash
dvc run -n datacheck \
        -d pipeline/04_check_data/conftest.py \
        -d pipeline/04_check_data/test_data.py \
        -d pipeline/01_data/preprocessing_data.csv \
        pytest pipeline/04_check_data -s -vv --reference_artifact pipeline/01_data/preprocessing_data.csv \
                                             --sample_artifact pipeline/01_data/preprocessing_data.csv \
                                             --ks_alpha 0.05

git add dvc.yaml dvc.lock
```

Now is time to update the remote repository, please run:

```bash
dvc push --remote s3remote
```





