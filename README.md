# Disaster Response Pipeline Project

## 1. Introduction

This project involves development of a supervised learning model to classify text messages (direct or via social media) that disaster response organizations receive following a disaster. Typically, different organizations/ teams within organizations focus on addressing different problems such as providing food, shelter, water, medical aid and/or products, fixing infrastructure. These problems have been mapped to 36 categories. Two datasets have been provided - one containing text messages received (and associated data) and another containing corresponding category labels have been provided for this project.    
 

## 2. Project Objectives

* Analyze text messages and devise a supervised learning model to identify problem classes/ categories
* Develop a client-server app using Python Flask to present an overview of the training data

## 3 Methodology 
* CRISP-DM (Cross-Industry Standard for Data Mining) https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining


## 4 Business Understanding
* Business Objective: To enable disaster response organizations to classify the incoming text messages received from diverse channels (direct and social media), into different (multiple) categories by analyzing the text message(s) to decode the problem(s) being reported so that they can effectively and efficiently channel their disaster response and relief efforts.

### 4.1 Business success criteria
* Model flags the problem areas (accurately predicts categories/ classes from the text messages) accurately to enable disaster response organizations to effectively and efficiently channel their disaster response and relief efforts


### 4.2 Define metrics to measure model performance 
* F1 score, precision and recall for the test set is presented for each category.
* Class membership would influence the selection of metric to measure model performance
  * For majority classes, presumably false positives would contribute an additional load (in terms of assistance required) in mostly an overloaded system in times of disaster. Hence precision (true positives amongst the predicted positives) would be an appropriate measure for such categories/ classes. The category "related" (which has a membership of around 76% of the dataset) falls in this group. A higher precision score implies superior model performance
  * For the remaining categories (which are in minority), a false negative would adversely affect disaster relief objectives. E.g. Messages in other categories including those requesting for water or medical assistance, fall in this group. In these cases, recall (true positives amongst total actual positives) measuring would be appropriate. A higher recall score implies superior model performance. However, where we need to balance both precision and recall (from cost of relief operation and risk to life and/or property), f1-score would be an appropriate performance measure. A higher f1 score implies superior model performance     


## 5 Data Understanding:

Word cloud of entire set of cleaned messages is shown below:

![Word cloud of entire set of cleaned messages](app/static/WordCloud - AllCleanedMessages.png)

Key Observations
* 6318 messages have no labels i.e. category is null for around 23% of the messages
* Around 21% of the messages have only one (amongst the 36) label
* There are messages with multiple labels - ranging from 2 to 23 labels; the number of such messages varies. The percentage of such messages varies from a very insignficant percentage (one or two messages with more than 20 classes) to about 14%



Key Takeaways
* This is an imbalanced dataset, irrespective of whether we consider each category/ label as a class, or each unique combination of labels as a class.


## 5 Data Preparation
### 5.1 Data Cleaning
Key steps performed in cleaning text messages
* Delete messages without any labels
* Normalize text i.e. convert to lower case
* Remove punctuation marks
* Expand short forms used in colloquial language (e.g.'ll -> will, 've -> have, n't -> to not, 're -> are)
* Remove stop words
* Convert words to their stemmed form using Snowball Stemmer


### 5.2 Data Integration and Formatting
* Each of the 36 categories/classes were converted to their respective binary classes, thus creating 36 binary variables representing corresponding categories.
* The message dataset and categories dataset containing the categories and their respective binary classes/ labels were concatenated into a single dataset and stored in  SQLite database

Steps 5.1 and 5.2 were completed using the code in process_data.py

### 5.3 Feature Engineering
Following feature extraction methods were adopted (after applying Lemmatization to cleaned text messages):
* Bag of Words 
* Term frequency - Inverse document frequency (TF-IDF) transformation
* Extract starting verbs 


## 6 Modeling
### 6.1 Identify and select modeling techniques
* This being a multi-label classification problem, MultiOutputClassifier and ClassifierChain were two obvious choices; the usage of former was recommended.
* Given the complexity of the problem - multi-label classification (which inherently leads to imbalanced classes), ensemble methods tend to relatively outperform other classifers; hence Random Forest Classifer was chosen. Besides, Random Forest algorithm also has a history of performing quite well in an imbalanced dataset scenario.

### 6.2 Train-test split
* Train-test split was 0.2 i.e. 80% for training and 20% for test


### 6.3 Model building
* sklearn's Pipeline method was used for feature extraction, feature union and model building
* sklearn's GridSearch method was used for exploring models built with different hyper parameter values and selection of best model; number of folds was limited to 2 (to minimize the model building cycle time)
* The hyper parameter space explored is given below:
  * features__text_pipeline__tfidf__use_idf   : True, False
  * features__text_pipeline__vect__ngram_range: (1,1) , (1,2)
  * features__text_pipeline__vect__max_df     : 0.5, 0.75, 1.0
  * features__text_pipeline__vect__max_features: None, 5000
  * feature_transformer_weights               : ({'starting_verb':0.5 ,'text_pipeline' : 1 }
                                                , {'starting_verb':1  ,'text_pipeline' : 0.5 }
                                                , {'starting_verb':1 ,'text_pipeline' : 0.8 })
  * clf__estimator__n_estimators              : 50
  * features__transformer_weights             : 3    


### 6.4 Model selection
* The best model resulting from the GridSearch on the above-mentioned hyper-parameter space, has the following parameter values:
features__text_pipeline__tfidf__use_idf       : False
  * features__text_pipeline__vect__ngram_range: (1,2)
  * features__text_pipeline__vect__max_df     : 0.75
  * features__text_pipeline__vect__max_features: 5000
  * feature_transformer_weights               : ({'text_pipeline' : 0.5, 'starting_verb':1})
  * clf__estimator__n_estimators              : 50
  * features__transformer_weights             : 3

## 7 Model Performance Evaluation
### 7.1 Evaluate performance and analyze test results
 
<details>
 <summary>Overall model performance</summary>
 
 Type of average | Recall | F1-Score
 ----------------|--------|---------
 Micro | 0.64 | 0.72
 Weighted | 0.64 | 0.67
 
</details>
<details>   
  <summary>Label-wise model performance</summary>
  * The F1 scores are very low for labels where number of messages are extremely low i.e. the extreme imbalance adversely affects model performance
  
  <img alt="Label-wise Model Performance" src="https://github.com/pravin096/dr-using-nlp/blob/main/web_app/app/static/LabelwiseModelPerformance.png">
  
  * However, there is a wide variance in the F1 scores for labels with similar levels of imbalance. This indicates that the quality of message content (linguisitc clarity) differs across labels.
  
  <img alt="Label-wise Model Performance" src="https://github.com/pravin096/dr-using-nlp/blob/main/web_app/app/static/MsgCountvsF1.png">
  
  
</details> 


### 7.2 Identify next steps
Inorder to improve the model performance further following measures could be explored:
i)  Stratified train-test split - This would ensure that the imbalance could be applied uniformly to train and test subsets, thus model training and model testing would also happen in a scenario wherein degree of imbalance is uniform across train and test

ii) Data augmentation through SMOTING - either oversampling the minority classes or undersampling the majority classes

iii) Application of class weights - sklearn offers class weight calculation method. This involves application of higher weights to minority classes (thus penalizing the error function in proportion to the class' imbalance). This is an alternative to (ii) above

iv) Application of sample weights - Similar to (iii) but the weights are applied to the samples ('X' part of the equation). This is an alternative to (ii) and (iii) above.

v) Reviewing the cleaned messages and applying further cleansing mechanisms using regex. Message translation also may be required in certain cases, as I came across some messsages in Hindi (expressed using English alphabets)! 


## 8 Deployment



## 9 Package Requirements and Operating Instructions
### 9.1 ETL Pipeline component
Component Filename: process_data.py

Package requirements:
* nltk
* numpy
* pandas
* sqlite3
* regex
* sqlalchemy

Operating Instructions:

* Run the following command in the project's root directory to set up your database
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

### 9.2 ML Pipeline component
Component Filename: train_classifier.py

Package requirements:
* joblib
* matplotlib
* nltk
* numpy
* pandas
* pickle
* plotly (including graph_objects)
* sqlite3
* sqlalchemy
* sklearn
* wordcloud

Operating Instructions:
* Run the following command in the project's root directory to set up your model.
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


### 9.3 App component
Component Filename: run.py

Package requirements:
* flask
* joblib
* nltk
* numpy
* pandas
* plotly (including graph_objs)
* sqlite3
* sqlalchemy
* sklearn

Operating Instructions:

* Run the following command in the app's directory to run your web app.
    `python run.py`

* Go to http://0.0.0.0:3001/
