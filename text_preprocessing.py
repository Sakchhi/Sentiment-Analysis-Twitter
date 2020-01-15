import json
import re

import pandas as pd
from nltk.corpus import stopwords

with open('contraction_map.json') as f:
    contraction_map = json.load(f)


def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text, flags=re.DOTALL)
    return input_text


def expand_contractions(text, contraction_mapping=contraction_map):
    # TODO: understand it
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def cleaning_text(text, html_pattern, stopwords_list):
    text = remove_pattern(text, html_pattern)
    text = remove_pattern(text, '#')
    text = expand_contractions(text)
    text = re.sub(r'[^a-zA-Z ]+', '', text)
    text = text.lower()
    text = ' '.join([w for w in text.split() if w not in stopwords_list])
    return text

    # if __name__ == '__main__':
    df_train = pd.read_csv('Data/train.csv')
    # df_test = pd.read_csv('Data/test.csv')
    print(df_train.label.value_counts())

    html_pattern = r'https*://[a-zA-z_.0-9/]+/* *'
    stop_words = set(stopwords.words('english'))

    df_test['cleaned_tweet'] = df_test.tweet.apply(lambda r: cleaning_text(r, html_pattern, stop_words))

    # TODO split words in hashtags
    # TODO parse emoticons
    # print(re.sub(r'[^\w\s]', '', df_train.iloc[i].cleaned_tweet), end='\n\n')
    # for i in range(10):
    #     print(df_train.iloc[i].cleaned_tweet, end='\n\n')
    df_test.to_excel("Data/train_cleaned_v0.xlsx", index=False)
