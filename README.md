# Claims-Knowledge-Graph
Here I classify existing claims in the Claimskg knowledge graph to True/False/Other claims.

I've uploaded the csv file of the graph in the directory and it is named "claimskg_result.csv". I used the web interface explorer in this [address](https://data.gesis.org/claimskg/explorer/home). 

## Project Set Up and Installation

We should install the required packages. I ran my whole project on my Google Colab and used its GPU especially for preprocessing the data. In this project I've upgraded the NumPy package, I've installed AutoGluon that can use its tabular predictor at the end of the program for benchmarking. I've installed keybert to use it for keyword extraction and filling the NANs and, installed seaborn to use it for drawing plots and some other packages that you can see in the very first cell of my program. 

## Dataset


I extracted the true, false, and mixture claims using ClaimsKG's web interface [Claim Explorer](https://data.gesis.org/claimskg/explorer/home). There are 27588 goal claims (4404 True, 12350 False, 10834 Mixture).

Because this explorer export the results to a CSV form just for a maximum of 10000 rows, I exported each of the true, false, and mixture records in separate CSV files. Because the distribution of all types of claims is not uniform, and both of the false and mixture claims have more than 10000 records, we lost 3184 claims to extract. But in comparison with the data set size and also, regarding a low difference between the number of total false and mixture claims, I guess it's not going to make a mistake.
The data is extracted from Claim Explorer are contained in three files:

* claimskg_false.csv - contains 10000 False claims and metadata about each claim (text, date, author, keywords, source, etc.)
* claimskg_mixture.csv - contains 10000 Mixture claims and metadata about each claim (text, date, author, keywords, source, etc.)
* claimskg_true.csv - contains 4404 True claims and metadata about each claim (text, date, author, keywords, source, etc.)


Each of these 3 csv files contains below features:

* id (object) - Claim link
* text (object) - Claim text
* date (object) - Claim published date
* truthRating (int64) - 1:False, 2: Mixture, 3: True - Label of classified claim
* ratingName (bool) - False, Mixture and True claim rating names
* author (object) - Name of the author
* headline (object) - Headlines of the claim
* named_entities_claim (object) - every cell contains an instance of a claim
* named_entities_article (object) - every cell contains an instance of an article
* keywords (object)- Keywords used in the claim
* source (object) - Claim reviewer source name
* sourceURL (object) - Claim reviewer source link
* link (object)- Link to the claim review
* language (object) - Language of the claim

**Note:** You may need to know this dataset is extracted from a knowledge graph containing all the relations between claims and their features. Some of the relations are one-to-many and some other are many-to-many.

## Problem Statement and Machine Learning Pipeline

I'm solving two problems on claims classification of the ClaimsKG dataset.

1. I'm facing a classification problem to classify each claim to be False, Mixture, or True, and, because I have the label of each claim, I should do a type of supervised learning. A good prediction will help us to precisely estimate whether a claim with specific metadata would be reviewed by a special reviewer as true, false, or a mixture of true and false (in mixture case some parts of the claim are true and the other parts would be false). It has benefits spreading the inference among the society and avoiding politicians and other public speakers to say wrong information in their talks.  

2. I should classify claims to be in two classes: {TRUE or FALSE} vs. {MIXTURE}. In this case, by getting a high score prediction from our models, we would know how much our estimator can declare a boundary between definite true/false claims and mixture claims reviewed by resources.

In both of the above problems, I'm using machine learning models to train classification models that their prediction indicates:

**Based on metadata and the text of a claim, which of the classes the claim belongs to.**
 
For solving this problem I first created a comprehensive table from all other 3 data tables of False/Mixture/True that I was extracted from ClaimsKG's web interface [Claim Explorer](https://data.gesis.org/claimskg/explorer/home). Then I have replaced the NaN cells with selections from unigrams or bigrams, Lemmatized and stemmed the words in the whole table, and removed the stop words using nltk library. At the last preprocessing step, I vectorized the whole table based on each class set of unigram and bigram choices, using some NLP metrics, and then used it as my data source. The source table contains all the claims important features in float amounts. Each float amounts indicate the feature's TF-IDF (Term Frequency and Document Frequency) based on the claim label. For example, I did a TF-IDF vectorization on the whole selections of the single words and unigrams that are existed in the True labeled claims and I did this job on the two other claim classes. Finally, I've merged all the three class vectorized table and have gotten the source table for training a classification model on it. A big part of this project consists of:

* Cleansing data and extracting information
* Filling the NaN cells of the table of the whole claims with appropriate n-grams
* Stemming, lemmatizing, removing the stop words
**Note: For filling the NaN cells of the named_entities_article column, I've Used unigram and bigrams from the headline and named_entities_claim to recognize the top 2 article topics. Because the number of NaNs in this column is more than 9000 and the number of words, especially in the named_entities_claim column is too much and, because we are searching for both single words and 2-grams, it's the bottleneck of my program to fill the NaNs of this column. Not using the Colab GPU, it takes about 40 minutes for running.

Offer: We can use only single words for article named-entity recognition.**
* Vectorizing the table and getting appropriate numerical feature columns
* Concatenating columns and creating neat tables
* Merging all the tables to one comprehensive table that we can use as a source for our model training  

When my source table is ready then we should solve the main problem. We should do the following jobs to get the results:

* Training a Scikit_Learn Random Forest model based on the given hyperparameters (number of estimators and leaves)
* Fitting a Multinomial Naive Bayes Classifier
* Using Support Vector Machine Classifier (SVC) model with various kernels like Linear, Polynomial, and, Gaussian
* Fitting K-Nearest Neighbors Classifier on the source data and making predictions
* Evaluating all models on some suitable classification metrics like f1-score, recall, and precision
* Training the dataset with the auto-ML library Autogluon Tabular predictor to find a benchmark for our model
* Evaluating the model scores of this auto-ML predictor and comparing to our results 
* Finding the best model for predicting our data 

The main parts of the implementation can be split into 3 steps:

1. Cleansing data, modifying the tables and merging them to create a neat textual table
2. Vectorizing the data to create a table of numerical features
3. Train a random forest and several other models (KNN, SVC, etc.) and evaluate the models on the test data using our metric scores
4. Using an AutoGluon Tabular predictor to compare our model metric scores with

## License

I have attached a Apache License 2.0 to this repository, click to see the license [here](https://github.com/EnsiyehRaoufi/Claims-Knowledge-Graph/blob/a86499380ddc48b24b1983e789d20a6c677eb938/LICENSE)

