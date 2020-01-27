import csv
import os

import pandas as pd
from sklearn import metrics
from textblob import TextBlob

import config
import run_config


def get_predictions(test_data):
    polarity = [TextBlob(i).sentiment.polarity for i in test_data]
    predictions = [1 if n > 0 else 0 for n in polarity]

    return predictions


if __name__ == '__main__':
    df_raw = pd.read_excel(os.path.join(config.DATA_DIR,
                                        "processed/train/preprocess/{}_full_cleaned_v0.10.xlsx").format(
        run_config.model_date_to_read, run_config.model_version_to_read))
    print(df_raw.columns.tolist())
    df_raw.lem_tweet.fillna('', inplace=True)
    # df_feature = pd.DataFrame(df_raw.lem_tweet.values, columns=['text'])
    # df_feature['label_rating'] = df_raw.label.apply(lambda r: int(r > 3))
    # print(df_feature.label_rating.value_counts())
    # X_train, X_val, y_train, y_val = train_test_split(df_feature.text, df_feature.label_rating, test_size=0.2,
    #                                                   random_state=42)

    predictions = get_predictions(df_raw.lem_tweet)
    y_val = df_raw[~df_raw.label.isnull()].label
    y_pred = predictions[:len(y_val)]
    print(y_val)

    accuracy_score = metrics.accuracy_score(y_val, y_pred)
    print(accuracy_score)

    cm = metrics.confusion_matrix(y_val, y_pred)
    tp, fn, fp, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = fp / (fp + tn)
    print("FPR = {}".format(fpr))
    print("TPR = {}".format(tp / (tp + fn)))

    f1 = metrics.f1_score(y_val, y_pred)
    print("F1 Score = {}".format(f1))

    columns = ['Run', 'Accuracy', 'FPR', 'F1 Score', 'Preprocessing', 'Feature', 'Model', 'Notes']
    preprocessing_notes = ""
    feature_notes = ""
    model_notes = "TextBlob baseline"
    misc_notes = ""
    fields = [run_config.model_version_to_write, accuracy_score, fpr, f1,
              preprocessing_notes, feature_notes, model_notes, misc_notes]
    with open(os.path.join(config.LOGS_DIR, r'results_summary.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    df_predictions = pd.DataFrame({"Predictions": y_pred, "Labels": y_val}, index=df_raw.loc[X_val.index]['id'])
    df_predictions.to_excel(os.path.join(config.OUTPUTS_DIR,
                                         '{}_textblob_baseline_v{}.xlsx'.format(run_config.model_date_to_write,
                                                                                run_config.model_version_to_write)))
