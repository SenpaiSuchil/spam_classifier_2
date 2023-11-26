from mail_tools import *
from data_tools import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.models import load_model
from data_save import *
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Cargar el modelo desde el archivo
print("ingrese el nombre completo (sin extension) del modelo que desea cargar:")
name=input()
model = load_model(f"saved_models\\{name}.keras")
#NN_data=joblib.load(f"saved_models\\{name}.joblib")

#directory where the mails are stored
spam_dir="spamtest\\"
ez_ham_dir="NOspamTest\\"
#model_test1_inputsize_65008

#data extraction
spam=mail_extractor(spam_dir)
ez_ham=mail_extractor(ez_ham_dir)


X,labels=mail_labeler(array1=spam,array2=ez_ham)#labeling the data with 1 for spam, and 0 for ez_ham

# labels=labels.T

# X_without_sw=[remove_stopwords(correo) for correo in X]#deleting all the stopwords from the mails

# X_tokenized,max_len,word_index=tokenizer(X_without_sw)#now we create a token representation of our data

# X_padded=padding(sequences=X_tokenized,max_len=NN_data.max_len)#fill the space to regularize the size of our inputs

#del X,X_without_sw,X_tokenized#deleting variables to save space in ram

X_train, X_temp, y_train, y_temp = train_test_split(X, labels.T, test_size=0.9, random_state=42)#split the data for training, validation and testing
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

vectorizer = CountVectorizer(stop_words='english', max_features=7000)
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

X_train_count=X_train_count.toarray()
X_test_count=X_test_count.toarray()

predictions = model.predict(X_test_count)

loss, accuracy = model.evaluate(X_test_count, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')


predicted_classes = (predictions > 0.5).astype('int32')

conf_matrix = confusion_matrix(y_test, predicted_classes)
print('Matriz de Confusión:')
print(conf_matrix)


class_report = classification_report(y_test, predicted_classes)
print('Reporte de Clasificación:')
print(class_report)