import itertools
import json
import os
import re

import pandas as pd
import wordninja
from expand_sms_slang import translator
from lemmatize import lemmatize_df

import config
import run_config

with open(os.path.join(config.UTILITIES_DIR, 'contraction_map.json')) as f:
    contraction_map = json.load(f)

with open(os.path.join(config.UTILITIES_DIR, 'stop_words.txt'), 'rb') as f:
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
    """
        Credits: DipanjanS
    :param text:
    :param contraction_mapping:
    :return:
    """
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


def camel_case_split(str):
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str)


def cleaning_text(text, pattern_dict, stopwords_list=stop_words):
    swear_regex = r'$&@*#'
    text = text.replace('$&@*#', 'shit')
    text = re.sub(pattern_dict["html_regex"], '', text)
    text = re.sub(pattern_dict["twitter_images_regex"], '', text)
    text = re.sub(pattern_dict["user_name_regex"], '', text)
    text = text.replace("#", "")
    split_regex = r'[A-Z]?[a-z]+[A-Z][a-z]*'
    pattern = re.compile(split_regex)
    text = ' '.join([r if not pattern.match(r) else ' '.join(camel_case_split(r))
                     for r in text.split()])
    text = translator(text)
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    text = expand_contractions(text)
    text = re.sub(r'[^a-zA-Z ]+', '', text)
    text = text.lower()
    text = ' '.join([w for w in text.split() if (w not in stopwords_list)
                     ])
    text = ' '.join([' '.join(wordninja.split(word)) for word in text.split()])
    return text


if __name__ == '__main__':
    df_train = pd.read_csv(os.path.join(config.DATA_DIR, 'raw/train.csv'))
    df_test = pd.read_csv(os.path.join(config.DATA_DIR, 'raw/test.csv'))
    df_full = df_train.append(df_test, ignore_index=True)
    # df_full = df_train.copy()
    print(df_full.shape)
    # print(df_full.tail())
    # print(df_train.label.value_counts())

    regex_list = {
        "html_regex": r'https*://[a-zA-z_.0-9-_/=&?]+/* *',
        "user_name_regex": r'@[A-Za-z0-9_]+',
        "twitter_images_regex": r'pic.twitter.com/[a-zA-Z0-9]+'
    }
    # stop_words = set(stopwords.words('english'))

    df_full['cleaned_tweet'] = df_full.tweet.apply(lambda r: cleaning_text(r, regex_list))
    # TODO split words in hashtags
    # TODO parse emoticons

    # for i in range(10):
    #     print(df_train.iloc[i].cleaned_tweet, end='\n\n')
    df_lem = lemmatize_df(df_full)
    # for i in range(10):
    #     print(df_full.lem_tweet.iloc[i], end='\n\n')
    df_full.to_excel(os.path.join(config.DATA_DIR, "processed/train/preprocess/{}_full_cleaned_v{}.xlsx".format(
        run_config.model_date_to_write, run_config.model_version_to_write)), index=False)
