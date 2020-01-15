import pickle

import pandas as pd
from sklearn.externals import joblib

count_vec_model = joblib.load('models/bag_of_words_v1.joblib')
df_test = pd.read_excel('Data/test_cleaned.xlsx')
bag_of_words_test = count_vec_model.transform(df_test.tweet)

test_bow = pd.DataFrame(bag_of_words_test.toarray(), columns=count_vec_model.get_feature_names())
# print(test_bow.head())

test_bow.to_csv('Data/bag_of_words_v1_test.csv')
mnb_model = pickle.load(open('models/20200115_mnb_bow_V1.pickle', 'rb'))

y_pred = mnb_model.predict(test_bow)

df_pred = pd.DataFrame(y_pred, index=df_test.id)
df_pred.to_csv('outputs/mnb_bow_V1.csv')
