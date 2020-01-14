import json
import re

import pandas as pd

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


def cleaning_hashtage(text):
    re.sub('#')


def text_cleaning(text):
    pass


if __name__ == '__main__':
    df_train = pd.read_csv('Data/train.csv')
    # print(df_train.label.value_counts())

    html_pattern = r'https*://[a-zA-z_.0-9/]+/* *'
    df_train['cleaned_tweet'] = df_train.tweet.apply(lambda r: remove_pattern(r, html_pattern))
    df_train['cleaned_tweet'] = df_train.cleaned_tweet.apply(lambda r: remove_pattern(r, '#'))
    df_train['cleaned_tweet'] = df_train.cleaned_tweet.apply(lambda r: expand_contractions(r))
    df_train['cleaned_tweet'] = df_train.cleaned_tweet.apply(lambda r: remove_pattern(r, r"\W+"))
    # TODO split words in hashtags
    # TODO parse emoticons
    # print(re.sub(r'[^\w\s]', '', df_train.iloc[i].cleaned_tweet), end='\n\n')
    for i in range(5):
        print(df_train.iloc[i].cleaned_tweet, end='\n\n')
