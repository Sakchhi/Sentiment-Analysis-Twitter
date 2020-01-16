import itertools
import json
import re

import pandas as pd

from expand_sms_slang import translator

with open('contraction_map.json') as f:
    contraction_map = json.load(f)

with open('stop_words.txt', 'rb') as f:
    stop_words = []
    for line in f:
        stop_words.append(line.decode("utf-8").strip())


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


def cleaning_text(text, pattern_dict, stopwords_list=stop_words):
    text = remove_pattern(text, pattern_dict["html_regex"])
    text = remove_pattern(text, pattern_dict["user_name_regex"])
    text = remove_pattern(text, '#')
    text = translator(text)
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    text = expand_contractions(text)
    text = re.sub(r'[^a-zA-Z ]+', '', text)
    text = text.lower()
    text = ' '.join([w for w in text.split() if w not in stopwords_list])
    return text


if __name__ == '__main__':
    df_train = pd.read_csv('Data/train/train.csv')
    df_test = pd.read_csv('Data/test/test.csv')
    df_full = df_train.append(df_test, ignore_index=True)
    print(df_full.columns, df_full.shape)
    print(df_full.tail())
    print(df_train.label.value_counts())

    regex_list = {
        "html_regex": r'https*://[a-zA-z_.0-9/]+/* *',
        "user_name_regex": r'@[A-Za-z0-9_]+'
    }
    # stop_words = set(stopwords.words('english'))

    df_full['cleaned_tweet'] = df_full.tweet.apply(lambda r: cleaning_text(r, regex_list))
    # TODO split words in hashtags
    # TODO parse emoticons
    # print(re.sub(r'[^\w\s]', '', df_train.iloc[i].cleaned_tweet), end='\n\n')
    # for i in range(10):
    #     print(df_train.iloc[i].cleaned_tweet, end='\n\n')
    df_full.to_excel("Data/train/full_cleaned_v0.4.xlsx", index=False)
