from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def stem_row(row):
    return ''.join([stemmer.stem(token) for token in row])


def stem_df(df):
    # df_raw = pd.read_excel(os.path.join(config.DATA_DIR, 'processed/train/preprocess/{}_full_cleaned_v{}.xlsx'.format(
    #     run_config.model_date_to_read, run_config.model_version_to_read)))
    df.cleaned_tweet.fillna('', inplace=True)
    # print(df_raw.columns)

    df['lem_tweet'] = df.cleaned_tweet.apply(lambda r: stem_row(r))
    return df


if __name__ == '__main__':
    stem_df()
