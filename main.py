from mail_tools import *
from data_tools import *
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from data_save import *
import joblib
from sklearn.feature_extraction.text import CountVectorizer


#directory where the mails are stored
spam_dir="spam\\"
ez_ham_dir="easy_ham\\"

#data extraction
spam=mail_extractor(spam_dir)
ez_ham=mail_extractor(ez_ham_dir)


X,labels=mail_labeler(array1=spam,array2=ez_ham)#labeling the data with 1 for spam, and 0 for ez_ham
#X,labels=mail_spam(array1=spam)#labeling the data with 1 for spam, and 0 for ez_ham



# X_without_sw=[remove_stopwords(correo) for correo in X]#deleting all the stopwords from the mails

# X_tokenized,max_len,word_index=tokenizer(X_without_sw)#now we create a token representation of our data


# X_padded=padding(sequences=X_tokenized,max_len=max_len)#fill the space to regularize the size of our inputs

#del X,X_without_sw,X_tokenized#deleting variables to save space in ram

#print(X_padded.T.shape)
#print(labels.T)

#print(len(max(X_padded,key=len)))

X_train, X_temp, y_train, y_temp = train_test_split(X, labels.T, test_size=0.9, random_state=42)#split the data for training, validation and testing
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#bolsa de palabras test:
vectorizer = CountVectorizer(stop_words='english', max_features=7000)
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

X_train_count=X_train_count.toarray()
X_test_count=X_test_count.toarray()


#########################
#parameters
epochs=200
batch_size=64

#embedding dimension
embedding_dim = 50

#vocabulary size
#vocab_size = len(word_index) + 1

#create the model
model = Sequential()

#add a embedding layer
# model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))

# #flatten the output of the previous layer
# model.add(Flatten())
model.add(Dense(1, activation='tanh', input_shape=(X_train_count.shape[1],)))
# add hidden layers
#model.add(Dense(1, activation='tanh'))
#model.add(Dense(1, activation='tanh'))

# output layer
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(X_train_count, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)



loss, accuracy = model.evaluate(X_test_count, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')


predictions = model.predict(X_test_count)
print(predictions)
predicted_classes = (predictions > 0.5).astype('int32')

conf_matrix = confusion_matrix(y_test, predicted_classes)
print('Matriz de Confusión:')
print(conf_matrix)


class_report = classification_report(y_test, predicted_classes,zero_division=1)
print('Reporte de Clasificación:')
print(class_report)

print("enter the model name:")
name=input()
# NN_data=NN_data(epochs,batch_size,embedding_dim,vocab_size,max_len)
model.save(f"saved_models\\model_{name}.keras")
# joblib.dump(NN_data,f"saved_models\\model_{name}_inputsize_{vocab_size}.joblib")