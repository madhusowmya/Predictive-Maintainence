# Predictive-Maintainence
Building a predictive model using machine learning to predict the probability of a device failure.
Data Science Case study
BACKGROUND A company has a fleet of devices transmitting daily sensor readings. They would like to
create a predictive maintenance solution to proactively identify when maintenance should be performed.
This approach promises cost savings over routine or time-based preventive maintenance because tasks
are performed only when warranted.
GOAL You are tasked with building a predictive model using machine learning to predict the probability
of a device failure. When building this model, be sure to minimize false positives and false negatives. The
column you are trying to Predict is called failure with binary value 0 for non-failure and 1 for failure.
Data Inference and Classification Process
1. Data is highly Imbalanced with a smaller number of features.
2. Based on the number of Sensors’ records, each day/each month. The number of Sensors failures pattern
is irregular. Therefore, a new column month is derived from date.
3. No Null values are present in the data.
4. Metric 7 and Metric 8 are Highly correlated.
5. Dropping one of the correlated columns and deleting the duplicates
6. Dimensionality reduction is not necessary for the given data.
7. Used stratified split to split the dataset into 80:20 ratio for train and test data respectively to
ensure an equal proportion of target classes in each split.
8. Encoded the categorical features using the one-hot encoder. Transformed the numerical features
using standard scalar.
9. Built the pipeline with the below steps as part of data transformation and model building.
Before training and testing the data, A Pipeline is created for Data Transformation, Model building.
a. Oversampling: Used the SMOTE techniques with sampling strategy of 0.1 to increase the
minority class samples to deal with imbalanced data
Under Sampling: Used Random Under Sampler to reduce the majority class, with sampling
strategy of 0.5
b. Estimator: Define the model.
10. To avoid overfitting K fold cross Validation is used with split of 10. The value for k is fixed to 10, a
value that has been found through experimentation to generally result in a model skill estimate with
low bias a modest variance.
11. A hyper parameter tuning grid.
12. GridSearchCV with the given model and transformed Data. It runs through all the different
parameters that is fed into the parameter grid and produces the best combination of parameters,
based on a scoring metric. The “best” parameters that GridSearchCV identifies are technically the
best that could be produced. The performance of the model is obtained from accuracy over train and
test data.
Model Building and Evaluation:
1. Used Logistic Regression to classify the target classes.
2. Included the cross-validation to avoid the over-fitting and grid-search for hyperparameter ( C:
regularization parameter) tuning.
3. Evaluated the model against test data. Observed ROC-AUC with mean scores. Since the data is highly
imbalanced Confusion matrix is useful. The performance of the model is listed below is evaluated based
on the Precision, Recall and F1 Score in the classification report.
I preferred the following Classifiers to predict class of the target variable. I am listing Pros and cons and
why I select that classifier.
1. LOGISTIC REGRESSION Classifier:
 Pros:
It is easy to implement, interpret and efficient to train. Only one categorical feature, When we
do one hot encoding, the number of dimensions increases, and there will be a good chance that data is
linearly separable, which leads to good accuracy.
Cons:
It may lead to overfitting if we have a lesser number of observations. There should not be
multicollinearity between independent variables.
2. Decision Tree Classifier:
Pros:
 It uses multiple algorithms to split each node, Features are selected automatically. Splits nodes on
all variables and select splits, that resulted in most homogeneity sub node. This method will be pure split
of each node.
Cons:
 It may lead to overfit the data.
3. Random Forest Classifier:
 Pros:
Random Forest is usually robust to outliers and can handle them automatically. Random Forest
algorithm is very stable. Even if a new data point is introduced in the dataset, the overall algorithm
is not affected much since the new data may impact one tree, but it is very hard for it to impact
all the trees.
Cons:
It is prone to overfitting. Low prediction accuracy compared to other Algorithms. Random Forest
is comparatively less impacted by noise
4. XGBoost Classifier:
Pros:
It is an efficient implementation of Gradient Boosting, where XGBoost, trees are built in parallel,
instead of sequentially like GBDT. XGBoost transforms the loss function into a more sophisticated
objective function containing regularization terms.
Cons:
It may lead to Overfit the data.
5. LGBoost Classifier:
Light GBM
 Pros:
It gives better accuracy among all other boosting algorithms. It takes leaf wise split approach
rather than a level-wise approach which leads to higher accuracy. It does not require heavy
feature engineering. In our case, we have one categorical feature have 12 classes which generate
several new features when we one-hot encode them. Using the Light GBM encoding step can be
avoided as it can handle categorical features as it is.
Since it is an ensemble technique, it gives better performance for imbalanced data.
Cons:
It is prone to overfitting which can be avoided by tuning hyperparameters like max-depth, early
stopping.
