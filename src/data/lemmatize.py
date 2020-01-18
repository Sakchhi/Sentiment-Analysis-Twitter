import spacy

nlp = spacy.load('en')


def spacy_lem(row):
    doc = nlp(row)
    return ' '.join([token.lemma_ for token in doc])


def lemmatize_df(df):
    # df_raw = pd.read_excel(os.path.join(config.DATA_DIR, 'processed/train/preprocess/{}_full_cleaned_v{}.xlsx'.format(
    #     run_config.model_date_to_read, run_config.model_version_to_read)))
    # df_raw.cleaned_tweet.fillna('', inplace=True)
    # print(df_raw.columns)

    df['lem_tweet'] = df.cleaned_tweet.apply(lambda r: spacy_lem(r))
    return df

    # for i in range(10):
    #     print(df_raw.cleaned_tweet.iloc[i])
    #     print(df_raw.lem_tweet.iloc[i], end='\n\n')


if __name__ == '__main__':
    lemmatize_df()
