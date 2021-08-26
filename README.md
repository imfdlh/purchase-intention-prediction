# Online Shoppers Purchasing Intention Project

The objective of this project is to classify whether a customer on e-commerce will bring revenue or not, simply says this project will build a classification model to predict whether a customer will end up shopping or not after going through the website.  In this project, there will be some model improvement to improve the performance. This project will also utilize a pipeline to integrate some of pre-processing steps and modeling steps.

# Deployment

You can try to predict some values here: https://fadilah-milestone2p1.herokuapp.com/

Template by Fadilah, powered by Bootstrap v5.1.

# Source

Obtained from UCI ML: <a href="https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset#"> link </a>

This dataset contains user activity on a e-commerce, there are 17 predictor and a target variable that determined whether the user end-up shopping or not after going through the website (simply bring revenue or not). Of the 12,330 sessions in the dataset, 84.5% (10,422) were negative class samples that did not end with shopping, and the rest (1908) were positive class samples ending with shopping.

The dataset consists of 10 numerical and 8 categorical attributes. The 'Revenue' attribute can be used as the class label.

The dataset consists of feature vectors belonging to 12,330 sessions. The dataset was formed so that each session would belong to a different user in a 1-year period to avoid any tendency to a specific campaign, special day, user profile, or period.

# Insights

You can also see the analysis from here: <a href="https://fadilah-milestone2p1.herokuapp.com/insights"> link </a>

-	For each customer, If the BounceRates increase then the ExitRates will follow and vice versa. This means, if a customer just leaves the page without doing anything, the ExitRates for that customer will increase.
-	If a customer has a higher PageValue compared to others, this customer will more likely to bring revenue because they have a high tendency to do shopping.
-	Most of the customer that will end up buying things in the session is the returning visitor. The customer who has been on the website is more likely to buy things.

**Recommendation**

-	Offer vouchers to those that have been browsing a lot of similar keywords or clicking through different pages during a session.
-	Offer a special voucher for the returning visitor once they marked several similar items, or revisit the items that they’ve visited before to boost their buying urgency.

# Preprocessing

1.	Convert target variable to numerical format.
2.	Feature encoding for nominal and cyclical features. For nominal will use one-hot encoding and for cyclical data will use sin cos transformation.
3.	Outlier treatment using several steps, such as logarithmic transformation, cap the outlier using Tukey method.
4.	Binning the least frequent value of a nominal variable that has been previously encoded to some labels. After that, encode the values using a one-hot encoder.
5.	Split the dataset into 70% training, 20% validation (separate validation set, different from the one in cross-validation), and 10% test set for model inference.
6.	Feature Standardization Scaling and SMOTE oversampling (sampling strategy = 1/3) that integrate inside the pipeline for each base model and best model.

# Evaluation

The evaluation metrics that will be the main concern in this project are AUC and Precision. The reason behind this decision is due to the nature of the data that has an imbalance class, thus the accuracy won’t represent the model's actual performance. As for Precision, we want to minimize the number of False Positives that happen. This is based on the consideration of which error will bring more harm towards the company, mostly revenue and ROI (Return on Investment). The result of this prediction model can be obtained will be utilized to decide whether to give a promotional voucher or coupon or discount or even rebate towards the customer than predicted to be ended up buying thins (bring revenue). The Redemption rate (the rate of costumer that redeemed a voucher/coupon) will be very low if we have a lot of False Positives, this will also affect the company's ROI since they have allocated their budget for promotion/making a campaign.

# Model Analysis

The best algorithm on the base model and best model (hyperparameter tuned) is consistent, which is Random Forest. In Random forest, the most important 10 features are:
-	PageValues, ExitRates, ProductRelated_Duration, ProductRelated, BounceRates, Administrative_Duration, Month_sin, Administrative, Month_cos, and TrafficType.

Except for PageValues, the score between other predictors doesn’t have a big difference, which means that all of them have similar contributions toward the model capability in predicting the target variable.

The main evaluation metrics for this project is AUC due to the class imbalance of the target variable. Besides that, the Precision score will also be a concern for this project evaluation based on our objective. If we compare the best score on cross-validation: validation set (test set) between the base model and the best model of the Random Forest, we get a small increase in AUC from 92.36% to 92.93%. For precision, unfortunately, there is a small decrease, from 68% to 67%. Besides the main evaluation metrics, we also successfully increase the F1 from 64% to 67% and Recall from 61% to 68%. These scores are obtained from the mean test score inside the cross-validation.

**Insights from Modeling**

Amongst the features/predictors, the 3 most important features are PageValues, ExitRates, and ProductRelated_Duration. These 3 features show how long they’ve been browsing several items on the website and the rate of how many people exit the page while they’re browsing. If we compare to EDA steps, this seems familiar as we previously have detected significant differences between the value for class 0 and class 1 in these features, especially PageValues and ExitRates. PageValues also has the smallest Gini impurity index since it is the most important feature compared to the other available features, even the score is much higher than the rest of the top important features. Also, if we take a look at the correlation matrix, this feature is the only one that has a medium correlation with the target variable.

**Best Model**

The best model obtained from the best model grid search by tuning them is Random Forest. The hyperparameters of the chosen model are as follows:
- max_depth=10
- min_samples_leaf=10
- The rest of the model hyperparameters are set to default.

Amongst the classifier, Random Forest tends to be more consistent and stable on all 4 evaluation metrics. The performance also seems to be okay on the new data since this model has an AUC of around 80% on a separate validation set and test set. This model has learned the rules underlying the data that make it can differentiate between class 1 and class 0 (between those customers who will end up buying and bring revenue and the ones that won't). This model also becomes less overfitted towards the training set after hyperparameter tuning has been done. Even so, there is some room for improvement towards this project, we can try another sampling strategy or more hyperparameter on Random Forest to be specific to have more differences between the training and validation performances.
