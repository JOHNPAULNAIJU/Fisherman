import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tokenizer = RegexpTokenizer(r'\w+')

print("Running...")

df  = pd.read_csv('/Library/MAIN/Wytty/Wytty ML/fisher-data.csv')

df = df[['dataframe_idx','Url','Subreddit','Tags']]
df = df.sample(frac=1).reset_index(drop='True')
df.isnull().sum()
df = df.dropna()
df['Subreddit'] = df['Subreddit'].apply(lambda x:x.lower())
df['Tags'] = df['Tags'].apply(lambda x:x.lower())
df['Url'] = df['Url'].apply(lambda x:tokenizer.tokenize(x))

givenIndices = [0,1,2,3]

s = df['Url']

indicesList = sorted(givenIndices, reverse=True)

for i in s:
    for indx in indicesList:
        if indx < len(i):
            i.pop(indx)

print("Removing from tags...")

df['Tags'] = df['Tags'].apply(lambda x:x.replace('_',''))
df['Tags'] = df['Tags'].apply(lambda x:x.split())
df['Subreddit'] = df['Subreddit'].apply(lambda x:x.split())
df['overall'] = df['Url'] + df['Subreddit'] + df['Tags']
df['overall'] = df['overall'].apply(lambda x:" ".join(x))
df['overall'] = df['overall'].apply(lambda x:x.lower())

ps = PorterStemmer()

print("Stemming...")

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

df['overall'] = df['overall'].apply(stem)

print("Vectorizing...")

vect = CountVectorizer(max_features=2000,stop_words='english')
vectors = vect.fit_transform(df['overall']).toarray()
similarity = cosine_similarity(vectors)