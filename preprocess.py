import re
import nltk # type: ignore
import string
from nltk.corpus import stopwords # type: ignore

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Initialize stopwords
stopword = set(stopwords.words("english"))

# Cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\s+|www\.\s+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    stemmer = nltk.SnowballStemmer("english")
    text = [stemmer.stem(word) for word in text]
    text = " ".join(text)
    return text
