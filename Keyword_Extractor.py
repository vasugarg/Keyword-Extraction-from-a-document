# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PyPDF2 as pdf
import nltk
import re 

# Loading the text
filename = 'JavaBasics-notes.pdf'
data = open(filename, 'rb')
read = pdf.PdfFileReader(data)
number_of_pages = read.getNumPages()
dataset = ""
for i in range(0, number_of_pages):
    page = read.getPage(i)
    i += 1
    dataset += page.extractText()

# Cleaning the data
# nltk.download('stopwords')
# nltk.download()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
review = re.sub('[^a-zA-Z]', ' ',dataset)
review = review.lower()
review = review.split()
review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]

# Weighing the Keywords
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(min_df = 0.002)
tfidf = tfidf_vectorizer.fit_transform(review)

feature_names = tfidf_vectorizer.get_feature_names()

# Arranging the keywords according to their frequency
feature_names = nltk.FreqDist(review)
keywords = feature_names.most_common(150)

# Displaying the list in csv file
import csv
with open('Keword_extraction - Sheet1.csv', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(keywords)
    
        
    
