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
    df_raw = pd.read_excel("Data/train/train_cleaned_v0.1.xlsx")
    print(df_raw.columns)

    count_vec_model, df_bow = get_bow(df_raw.cleaned_tweet.tolist())
    print(df_bow.head())
    pickle.dump(count_vec_model, open("models/bag_of_words_v0.1.pickle", 'wb'))
    # TODO JOBLIB vs PICKLE
    df_bow.to_csv('Data/train/bag_of_words_v0.1.csv', index=False)
