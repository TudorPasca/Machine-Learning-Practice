import nltk
import pandas as pd
import re

from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

df = pd.read_csv("movie_data.csv", encoding="utf-8")
print(df.head(3))
print(df.shape)

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()
def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

nltk.download('stopwords')
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot') if w not in stop])
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
small_param_grid = [
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer],
        'vect__use_idf': [False],
        'vect__norm': [None],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    }
]
lr_tfidf = Pipeline([
    ('vect', tfidf),
    ('clf', LogisticRegression(solver='liblinear'))
])
gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=2)
gs_lr_tfidf.fit(X_train, y_train)
print(f'Best params: {gs_lr_tfidf.best_params_}')