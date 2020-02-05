import csv
import os
import pickle

import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC

import config
import run_config


def get_predictions(train_data, train_labels, test_data):
    mnb_model = SVC()
    mnb_model.fit(train_data, train_labels)
    predictions = mnb_model.predict(test_data)

    pickle_file_name = os.path.join(config.MODEL_DIR, 'classifier_model/{}_svc_bow_v{}.pickle'.format(
        run_config.model_date_to_read, run_config.model_version_to_read))
    pickle.dump(mnb_model, open(pickle_file_name, 'wb'))
    return predictions


if __name__ == '__main__':
    df_bow = pd.read_csv(os.path.join(config.DATA_DIR, 'processed/train/feature_eng/{}_bag_of_words_v{}.csv'.format(
        run_config.model_date_to_read, run_config.model_version_to_read)))
    df_train = pd.read_csv(os.path.join(config.DATA_DIR, 'raw/train.csv'))
    # X_train, X_val, y_train, y_val = train_test_split(df_bow, df_train.label, test_size=0.2)
    print(df_bow.shape, df_train.shape)

    # df_data = df_bow.copy()
    # df_data['sent_label'] = df_train.iloc[:, 1].values
    # print(df_data.shape)
    # print(df_data.sent_label.value_counts())
    #
    # negative_label_indices = df_data[df_data.sent_label == 0].index
    # sample_size = sum(df_data.sent_label == 1)
    # random_indices = np.random.choice(negative_label_indices, sample_size, replace=False)
    # postive_label_indices = df_data[df_data.sent_label == 1].index
    #
    # under_sampled_indices = np.concatenate([postive_label_indices, random_indices])
    # df_under_sample = df_data.loc[under_sampled_indices]
    # print(df_under_sample.sent_label.value_counts())
    # print(len(negative_label_indices), sample_size, len(df_under_sample), len(under_sampled_indices))

    # y_pred = get_predictions(df_under_sample.iloc[:, :-1], df_under_sample.sent_label, df_bow)
    y_pred = get_predictions(df_bow, df_train.label, df_bow)
    print(len(y_pred), len(df_train.label))

    accuracy_score = metrics.accuracy_score(df_train.label, y_pred)
    print(accuracy_score)

    cm = metrics.confusion_matrix(df_train.label, y_pred)
    tp, fn, fp, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = fp / (fp + tn)
    print("FPR = {}".format(fpr))
    print("TPR = {}".format(tp / (tp + fn)))

    f1 = metrics.f1_score(df_train.label, y_pred)
    print("F1 Score = {}".format(f1))

    columns = ['Run', 'Accuracy', 'FPR', 'F1 Score', 'Preprocessing', 'Feature', 'Model', 'Notes']
    preprocessing_notes = ""
    feature_notes = ""
    model_notes = "SVM"
    misc_notes = ""
    fields = [run_config.model_version_to_write, accuracy_score, fpr, f1,
              preprocessing_notes, feature_notes, model_notes, misc_notes]
    with open(os.path.join(config.LOGS_DIR, r'results_summary.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    # df_predictions = pd.DataFrame({"Predictions": y_pred, "Labels": y_val}, index=df_raw.loc[X_val.index]['review_id'])
    # df_predictions.to_excel(os.path.join(config.OUTPUTS_DIR,
    #                                      '{}_TextBlob_Tfidf_2gram_10K_v{}.xlsx'.format(run_config.model_date_to_write,
    #                                                                                    run_config.model_version_to_write)))
