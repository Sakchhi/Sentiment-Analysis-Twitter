import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def get_bow(data):
    count = CountVectorizer()
    count.fit(data)
    bag_of_words = count.transform(data)
    bow_df = pd.DataFrame(bag_of_words.toarray(), columns=count.get_feature_names())
    return count, bow_df


if __name__ == '__main__':
    df_raw = pd.read_excel("Data/train/full_cleaned_v0.1.xlsx")
    train_length = df_raw[~df_raw.label.isnull()].shape[0]
    print(df_raw.columns, train_length)

    count_vec_model, df_bow = get_bow(df_raw.cleaned_tweet.tolist())
    print(df_bow.head())
    pickle.dump(count_vec_model, open("models/bag_of_words_v0.1.pickle", 'wb'))
    # TODO JOBLIB vs PICKLE
    df_bow_train = df_bow.loc[:(train_length - 1)]
    df_bow_test = df_bow.loc[train_length:]
    print(df_bow_train.shape, df_bow_test.shape, df_raw.shape)
    df_bow_train.to_csv('Data/train/bag_of_words_v0.1.1.csv', index=False)
    df_bow_test.to_csv('Data/test/bag_of_words_v0.1.1.csv', index=False)
