
get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install nltk')


import numpy as np 
import pandas as pd   
import nltk 
nltk.download('stopwords')  
from nltk.corpus import stopwords 



reviews = pd.read_csv("Data/sentiment_analysis.csv")
reviews.head(10)



reviews.shape


# exploratory data analysis before model selection


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn 



seaborn.set_theme(style = "darkgrid")
seaborn.countplot(x = 'class', data= reviews);


# Data preprocessing and data cleaning


X = reviews['text'].values # feature vectors 
y = reviews['class'].values # labels


# using different regular expressions for text preprocessing


import re # regular expression python library
processed_reviews = []

for reviews in range(0, len(X)):  
    # Removing  all the special characters
    processed_review = re.sub(r'\W', ' ', str(X[reviews]))
 
    # remove all single characters
    processed_review = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_review)
 
    # Remove single characters from the start
    processed_review = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_review) 
 
    # Substituting multiple spaces with single space
    processed_review = re.sub(r'\s+', ' ', processed_review, flags=re.I)
 
    # Removing prefixed 'b'
    processed_review = re.sub(r'^b\s+', ' ', processed_review)
 
    # Converting to Lowercase
    processed_review = processed_review.lower()
 
    processed_reviews.append(processed_review)


# Using TF-IDF vectorizer for feature extractor 


from sklearn.feature_extraction.text import TfidfVectorizer  
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.6, stop_words=stopwords.words('english'))
# the indivisual word sould be in atleast 5 documets and atmost 60% of the documents
X = tfidfconverter.fit_transform(processed_reviews).toarray()


# Dividing the data into training and testing set 


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
 


# importing the text classification model from sklearn 

from sklearn.naive_bayes import MultinomialNB
text_classifier = MultinomialNB()  
text_classifier.fit(X_train, y_train)



# now passing the test set to make predictions
predictions = text_classifier.predict(X_test)


# Model evaluation and analysis


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))



# plotting the data
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(text_classifier, X_test, y_test);


# plotting the precision-recall curve for two class problem


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

disp = plot_precision_recall_curve(text_classifier, X_test, y_test)


# So we saw that average precision is 88%. Now we can change the parameters and models_selection to see how accuracy,
# precision and recall will behave.




