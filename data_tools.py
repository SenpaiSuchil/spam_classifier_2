from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

#if you are running this for the first time in your pc yo may want to execute this lines of code on python cmd or cli before the code execution:
#import nltk
#nltk.download('stopwords')
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)


def tokenizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_len=len(max(sequences,key=len))
    return sequences,max_len,tokenizer.word_index


def padding(sequences,max_len):

    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences



