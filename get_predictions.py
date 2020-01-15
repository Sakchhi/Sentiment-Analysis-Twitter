import pickle

import pandas as pd

if __name__ == '__main__':
    df_test = pd.read_csv('Data/test/test.csv')
    # html_regex = r'https*://[a-zA-z_.0-9/]+/* *'
    # # stop_words = set(stopwords.words('english'))
    #
    # with open('stop_words.txt', 'rb') as f:
    #     stop_words = []
    #     for line in f:
    #         stop_words.append(line.decode("utf-8").strip())
    #
    # df_test['cleaned_tweet'] = df_test.tweet.apply(lambda r: cleaning_text(r, html_regex, stop_words))
    # df_test.to_excel('Data/test/test_cleaned_v0.1.xlsx', index=False)
    #
    # count_vec_model = pickle.load(open('models/bag_of_words_v0.1.pickle', 'rb'))
    # df_test = pd.read_excel('Data/test/test_cleaned_v0.1.xlsx')
    # bag_of_words_test = count_vec_model.transform(df_test.cleaned_tweet)
    #
    # test_bow = pd.DataFrame(bag_of_words_test.toarray(), columns=count_vec_model.get_feature_names())
    # print(test_bow.head())
    #
    # test_bow.to_csv('Data/test/bag_of_words_v0.1.csv')
    mnb_model = pickle.load(open('models/20200115_mnb_bow_v0.2.pickle', 'rb'))

    test_bow = pd.read_csv('Data/test/bag_of_words_v0.2.csv')
    y_pred = mnb_model.predict(test_bow)

    df_pred = pd.DataFrame(y_pred, index=df_test.id, columns=['label'])
    df_pred.to_csv('outputs/20200115_mnb_bow_v0.2.csv')
