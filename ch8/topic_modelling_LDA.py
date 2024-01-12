import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df = df.rename(columns={"0": "review", "1": "sentiment"})
count = CountVectorizer(stop_words="english", max_df=0.1, max_features=5000)
X = count.fit_transform(df['review'].values)
lda = LatentDirichletAllocation(n_components=10, learning_method="batch", random_state=123)
X_topics = lda.fit_transform(X)
print(lda.components_.shape)
n_top_words = 5
feature_names = count.get_feature_names_out()
for topic_id, topic in enumerate(lda.components_):
    print(f"Topic {topic_id + 1}:")
    print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1: -1]]))
