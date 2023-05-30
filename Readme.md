# Fake News Detection using Machine Learning ,Deep Learning and Ensemble 
### [Read Paper](https://drive.google.com/file/d/1jVoJ4LMJ0iCCb1AvehDo_eJ9ErSRBiVR/view?usp=sharing)
We live in a world where everyone is connected to each other virtually through the use of social media.Social media sites and search engines serve as the major sources of information and news we consume. However, this ease and efficient way of communication also comes with major side effects. Social media is very helpful for the dissemination of news, but fake news spreads like wildfire on such platforms. This is because fake news articles have catchy headlines and content that grab people's attention and sway their opinions. To counter the spread of fake news by detecting it easily and at an early stage, we have conducted the following research.

The main goal of our paper is to propose an efficient combination of machine learning and deep learning classifiers that can detect fake news articles with sufficient accuracy to be considered trustworthy. We employ ensemble methods such as voting, bagging, and stacking to combine various machine learning and deep learning classifiers and achieve the highest possible accuracy. These ensemble techniques help increase the collective accuracy of our meta-model (created by combining various classifiers) by identifying useful patterns in the output from each fake news classifier.

Before delving into the prediction aspect, we first define what constitutes fake news for our research. Fake news, in our context, refers to any article that is not factual or has been identified as fake news by a verified news source. Our project utilizes the ISOT dataset and FakeNews dataset as the sources of our fake and true news samples. Both datasets have been manually sourced and verified and are commonly used in related studies on fake news detection.

In addition, we combine the two datasets to investigate the potential issue of overfitting. If overfitting exists, the prediction accuracy for the combined datasets would be significantly lower compared to the accuracies obtained when using the individual datasets alone. This discrepancy arises due to the diverse nature of the combined dataset, as the ISOT and FakeNews datasets originate from different sources and were collected at different points in time.

Each fake news classifier (Naive Bayes, Logistic Regression, SVM, CNN, LSTM, and BERT) is individually tested on each of the aforementioned datasets, and their respective accuracies are compared. We then explore various combinations of these classifiers using bagging, voting, and stacking techniques to determine the optimal combination of classifiers and ensemble methods that yield the highest accuracy in predicting fake news.

## Project Design:
<img src="https://github.com/jinnies47/FakeNewsDetection/assets/63533609/8bdfb725-f110-46bf-b328-72fa2c1b6b28" alt="project_design" width="500" height="800" >

## Installation
All are Google Collab files you can open them in Collab and run them.

## Datasets
 Data Description
Two datasets are used for testing and training the classifiers. Both of these are taken from the Kaggle repository,
the first one is the FakeNews dataset consisting of a collection of news articles. The second one is the ISOT fake news
dataset comprising of 2 different files i.e. True containing authentic news and Fake containing fake news.
1. FakeNews Dataset
The FakeNews Dataset comprises a collection of news articles along with associated metadata, including ID, title,
author, text, and a label indicating the article’s reliability. This dataset aims to facilitate the exploration and analysis of
news articles, with a particular focus on assessing their potential reliability. The dataset fields are described as follows:
* ID: A unique identifier assigned to each news article.
* Title: The title of the news article.
* Author: The author or authors credited with writing the article.
* Text: The textual content of the news article. It is worth noting that the text may be incomplete in some cases.
* Label: A categorical label indicating the reliability of the article. This label serves to categorise and analyse the
articles, particularly in tasks such as classification or identifying potentially unreliable news sources.
The Kaggle dataset provides researchers and analysts with a valuable resource for studying news articles, investi-
gating authorship, and performing text analysis. It can be utilised for various purposes, including text classification,
natural language processing, and developing machine learning algorithms focused on detecting potentially unreliable
news sources.
2. ISOT Dataset
The ISOT (Integrated Scraping of Online Text) dataset comprises two distinct files, each containing fake and real
news articles respectively. Each file includes information related to the title, text, subject, and date of the news articles.
The dataset fields are described as follows:
* Title: The title of the news article.
* Text: The textual content of the news article.
* Subject: The subject or topic to which the news article pertains.
* Date: The date when the news article was published or scraped from the source.
The ISOT dataset offers researchers and analysts a comprehensive collection of both fake and real news articles,
allowing for the examination of fake news characteristics and facilitating comparisons with genuine news articles. It
can be employed for various purposes, such as text analysis, news verification, and training machine learning models
to differentiate between fake and real news sources. The inclusion of subject and date information further enables
temporal and topical analysis of the articles

## Performance Evaluation
![image](https://github.com/jinnies47/FakeNewsDetection/assets/63533609/1eee02a5-46e2-41a3-9c73-b8717db68a83)

To further improve the accuracy of our classifiers across the three datasets, we propose the utilization of ensemble
methods such as Stacking, Bagging, and Majority Voting. These ensemble techniques have shown promising results in
combining the predictions of multiple classifiers and can potentially enhance the overall performance and robustness
of our models.
By employing these ensemble methods, we expect to leverage the diverse strengths of individual classifiers and
mitigate their weaknesses. This ensemble approach can potentially lead to improved accuracy and more reliable fake
news detection across different datasets.


## The Experiment Results  show that
### Bagging Ensemble
* The Combination of BERT+CNN+NB gives highest accuracy of **97.43%** for ISOT dataset
* The Combination of BERT+CNN+LSTM gave the highest accuracy of **91.49%** for FakeNews Kaggle dataset.
* For the Combination of the ISOT and FakeNews Kaggle dataset and the combination of BERT+CNN+SVM gives the highest accuracy of **95.54%.**
 This implies a bagging ensemble with BERT, CNN and either or all of SVM, NB and LSTM can be the most efficient
combination to classify fake news. This is also verified incase of bagging with a combination of 5 classifiers where 
BERT, CNN, SVM, NB and LSTM combination gives highest accuracy for our two datasets and their combination.

### Voting Ensemble
* The Combination of BERT+CNN+SVM give highest accuracy of **99.94%** for ISOT dataset 
* The Combination of BERT+CNN+LSTM gives highesr accuracy of **91.68%** for FakeNews Kaggle dataset 
*  BERT+CNN+LSTM gives highest accuracy of **96.52%** for the KAGSOT dataset (combination of ISOT and FakeNews Kaggle dataset).

This shows that for voting a combination of BERT+CNN+LSTM+SVM will yield the best accuracy for fake news classification which is verified
when compared to accuracy in using voting ensemble to combine BERT, CNN, LSTM and SVM which gives highest accuracy 
for our two datasets and their combination. 

### Stacking Ensemble
* The Combination of BERT + LSTM + Logistic Regression+Naive Bayes gives the highest accuracy of **99.89%** for the ISOT dataset 
* The Combination of BERT + LSTM + Logistic Regression+Naive Bayes gives the highest accuracy of **91.27%** for the FakeNews Kaggle datasets 
* SVM+CNN+LSTM gives highest accuracy of **96.52%** for the KAGSOT dataset (combination of ISOT and FakeNews Kaggle dataset).

Hence the combination of SVM ,CNN, LSTM, Logistic regression, Bert and
CNN can be the best combination of classifiers to detect fake news using stacking ensemble method. This can be
verified when we see the combination of SVM ,CNN, LSTM, Logistic regression, Bert and CNN which gives one of the highest accuracies for our two datasets and their combination.
## Results and Discussion
Overall voting gives a very high accuracy of **96.53%** when tested on the combination of the two datasets using the
classifiers BERT+CNN+SVM combination. This combination uses two ML and one DL based classifier hence we
can leverage the power of ML to give high acuuracy with small dataset and the high learning ability of DL classifiers.
Voting also gives a very high accuracy of **91.68%** when tested on the FakeNews Kaggle datasets using the classifiers
BERT+CNN+LSTM combination.
Stacking gives a very high accuracy of **99.98%** when tested on the ISOT dataset with classifiers BERT+CNN+SVM+LSTM
and Logistic Regression combination. This combination also uses ML and DL based classifier hence we can leverage
the power of ML to give high acuuracy with small dataset and the high learning ability of DL classifiers.



