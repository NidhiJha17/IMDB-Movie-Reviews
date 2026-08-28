import pandas as pd
import re 
import string 

#loading the data
data= pd.read_csv(r"C:\Users\nidhi\Downloads\IMDB Original dataset\IMDB Dataset.csv")

#review cleaning
def clean_text(review):
    review= review.lower()
    review= re.sub(r"https\s+", " ", review)
    review= re.sub(r"[^\w\s]", " ", review)
    review= review.strip()
    return review

data['clean_text']= data['review'].apply(clean_text)

#NLP Preprocessing pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

data['tokens']= data['review'].apply(lambda x:x. split())

stop_words= set(stopwords.words('english'))
data['tokens']= data['tokens'].apply(lambda words: [word for word in words if word not in stop_words])

lemmatizer= WordNetLemmatizer()
data['tokens']=  data['tokens'].apply (lambda words: [lemmatizer.lemmatize(word)for word in words])

data ['clean_text']= data['tokens'].apply (lambda x: " ". join(x))

#converting text to numerical features
from sklearn.feature_extraction.text import TfidfVectorizer
Vectorizer= TfidfVectorizer(max_features=5000)
x= Vectorizer.fit_transform(data['clean_text'])
y= data['oh_label'] 