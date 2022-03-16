# Model Card

Model cards are a succinct approach for documenting the creation, use, and shortcomings of a model. The idea is to write a documentation such that a non-expert can understand the model card's contents. For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Ivanovitch Silva created the model. A complete data pipeline was built using DVC and Scikit-Learn to train a Decision Tree model. For the sake of understanding, a simple hyperparameter-tuning was conducted, and they are described in a [yaml file](https://github.com/ivanovitchm/mlops_nd_c3/blob/main/params.yaml). 

## Intended Use
This model is used as a proof of concept for the evaluation of an entire data pipeline incorporating MLOps assumptions. The data pipeline is composed of the following stages: a) ``data``, b) ``eda``, c) ``preprocess``, d) ``check data``, e) ``segregate``, f) ``train``, g) ``evaluate`` and h) ``check model``.

## Training Data

The dataset used in this project is based on individual income in the United States. The *data* is from the *1994 census*, and contains information on an individual's ``marital status, age, type of work, and more``. The target column, or what we want to predict, is whether individuals make *less than or equal to 50k a year*, or *more than 50k a year*.

You can download the data from the University of California, Irvine's [website](http://archive.ics.uci.edu/ml/datasets/Adult).

<center>
<img width="600" src="../images/gender_race.png"> <img width="600" src="../images/gender_workclass.png">
</center>


## Evaluation Data

## Metrics
_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations

## Caveats and Recommendations
