# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Prediction task is to determine whether a person makes over 50K a year. The project implements a Random Forest Classifier with the following parameters:

n_estimators : 10
max_depth: None
min_samples_split:2

The model is saved as pickle file in model folder.
## Intended Use

The model can be used to predict the salary level of an individual based on the following attributes:
age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week and native-country.

The target variable is the income or salary.

## Training Data
The Census Income dataset contains information about individuals from the 1994 US Census database. It includes demographic features such as age, education, occupation, marital status, and more, as well as a binary variable indicating whether an individual's income exceeds $50,000 per year. The goal of the dataset is to predict whether an individual's income exceeds $50,000 per year based on the other features.

A 80-20 split was used to break this dataset into a train and test set. Stratification on target label "salary" was applied. To use the data for training a One Hot Encoder was used on the categorical features and a label binarizer was used on the target.

## Evaluation Data
20% of the dataset was set aside for model evaluation. Transformation was applied on the categorical features and the target label respectively using the One Hot Encoder and label binarizer fitted on the train set.
## Metrics
The classification performance is evaluated using precision, recall and fbeta metrics:

Precision: 0.7167207792207793, Recall: 0.5631377551020408, fbeta: 0.6307142857142857
## Ethical Considerations

Fairness and bias: There is a risk of bias in the dataset due to the way it was collected. For example, some groups of people may be underrepresented in the dataset, leading to biased predictions. This could result in unfair treatment of certain individuals or groups.

Privacy and confidentiality: The dataset contains sensitive information about individuals, such as their income, occupation, and education level. It is important to ensure that this information is not misused or shared in a way that could harm individuals.

Data quality and accuracy: There may be errors or inaccuracies in the dataset, which could lead to incorrect predictions. It is important to validate the data and ensure that it is of sufficient quality for the intended use.

Transparency and explainability: It is important to be transparent about how the dataset was collected, processed, and used. This can help build trust with users and stakeholders, and ensure that the predictions are fair and accurate.

Algorithmic accountability: The use of machine learning algorithms to make predictions based on the dataset raises questions of accountability. It is important to ensure that the algorithms are fair and transparent, and that their decisions can be explained and challenged if necessary.

## Caveats and Recommendations
Use under adult supervision.