import csv
import os

import pandas as pd
from keras.layers import Flatten
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn import metrics

import config
import run_config

df_raw = pd.read_excel(os.path.join(config.DATA_DIR, 'processed/train/preprocess/20200120_full_cleaned_v0.10.xlsx'))
print(df_raw.columns, df_raw.shape)

df_raw.lem_tweet.fillna('', inplace=True)
X = df_raw.lem_tweet
y = df_raw.label

X_train = df_raw[~df_raw.label.isnull()].lem_tweet
y_train = df_raw[~df_raw.label.isnull()].label

X_test = df_raw[df_raw.label.isnull()].lem_tweet

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 50
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open(os.path.join(config.ROOT_DIR, 'glove6b100dtxt/glove.6B.100d.txt'), encoding='utf8')

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())
history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
predictions = model.predict(X_test)
val_predict = model.predict(X_train)
val_predict = [1 if i > 0.5 else 0 for i in val_predict]

accuracy_score = metrics.accuracy_score(y_train, val_predict)
print(accuracy_score)

cm = metrics.confusion_matrix(y_train, val_predict)
tp, fn, fp, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
fpr = fp / (fp + tn)
print("FPR = {}".format(fpr))
print("TPR = {}".format(tp / (tp + fn)))

f1 = metrics.f1_score(y_train, val_predict)
print("F1 Score = {}".format(f1))

preds = [1 if i > 0.5 else 0 for i in predictions]
df_pred = pd.DataFrame(preds, index=df_raw[df_raw.label.isnull()].id, columns=['label'])
print(df_pred.head())
df_pred.to_csv(df_pred.to_csv(os.path.join(config.OUTPUTS_DIR, '{}_{}_v{}.csv'.format(
    run_config.model_date_to_write, "lstm_glove100d", run_config.model_version_to_write))))

columns = ['Run', 'Accuracy', 'FPR', 'F1 Score', 'Preprocessing', 'Feature', 'Model', 'Notes']
preprocessing_notes = "Keras Tokenizer in cleaned"
feature_notes = ""
model_notes = "Keras LSTM with glove100d"
misc_notes = ""
fields = [run_config.model_version_to_write, accuracy_score, fpr, f1,
          preprocessing_notes, feature_notes, model_notes, misc_notes]
with open(os.path.join(config.LOGS_DIR, r'results_summary.csv'), 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
