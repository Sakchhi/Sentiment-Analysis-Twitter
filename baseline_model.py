import pickle

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def get_predictions(train_data, train_labels, test_data):
    mnb_model = MultinomialNB()
    mnb_model.fit(train_data, train_labels)
    predictions = mnb_model.predict(test_data)

    pickle.dump(mnb_model, open('models/20200115_mnb_bow_v0.1.pickle', 'wb'))
    return predictions


if __name__ == '__main__':
    df_bow = pd.read_csv('Data/train/bag_of_words_v0.1.csv')
    df_train = pd.read_csv('Data/train/train.csv')
    X_train, X_val, y_train, y_val = train_test_split(df_bow, df_train.label, test_size=0.2)

    y_pred = get_predictions(X_train, y_train, X_val)

    accuracy_score = metrics.accuracy_score(y_val, y_pred)
    print(accuracy_score)

    cm = metrics.confusion_matrix(y_val, y_pred)
    tp, fn, fp, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print("FPR = {}".format(fp / (fp + tn)))
    print("TPR = {}".format(tp / (tp + fn)))

    f1 = metrics.f1_score(y_val, y_pred)
    print("F1 Score = {}".format(f1))
