import pickle

import pandas as pd
from nltk.corpus import stopwords

from text_preprocessing import cleaning_text

if __name__ == '__main__':
    df_test = pd.read_csv('Data/test/test.csv')
    html_pattern = r'https*://[a-zA-z_.0-9/]+/* *'
    stop_words = set(stopwords.words('english'))

    df_test['cleaned_tweet'] = df_test.tweet.apply(lambda r: cleaning_text(r, html_pattern, stop_words))
    df_test.to_csv('Data/test/test_cleaned_v0.xlsx', index=False)

    count_vec_model = pickle.load(open('models/bag_of_words_v0.pickle', 'rb'))
    # df_test = pd.read_excel('Data/test/test_cleaned_v0.xlsx')
    bag_of_words_test = count_vec_model.transform(df_test.cleaned_tweet)

    test_bow = pd.DataFrame(bag_of_words_test.toarray(), columns=count_vec_model.get_feature_names())
    # print(test_bow.head())

    test_bow.to_csv('Data/test/bag_of_words_v0.csv')
    mnb_model = pickle.load(open('models/20200115_mnb_bow_v0.pickle', 'rb'))

    y_pred = mnb_model.predict(test_bow)

    df_pred = pd.DataFrame(y_pred, index=df_test.id, columns=['label'])
    df_pred.to_csv('outputs/20200115_mnb_bow_v0.csv')
