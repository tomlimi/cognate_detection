import os

import transformers

from data_wrapper import ClassificationDataset

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

import xgboost as xgb
import fasttext
import numpy as np
from gensim.models import KeyedVectors

import logging

logging.basicConfig(level=logging.INFO)

LANGUAGES = ['en', 'fr', 'de', 'da', 'el', 'it', 'no', 'nl', 'af', 'br', 'ca', 'cs', 'cy', 'da', 'es',  'fy', 'ga', 'gd', 'gl', 'gv', 'is',
             'kw', 'la', 'lb', 'lt', 'lv', 'nb', 'nl', 'nn', 'no', 'oc', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv', 'wa']

CLASSES = ['der', 'cog', 'none']

fasttext_mapping = {'br': 'fr', 'cy': 'el', 'fy': 'nl', 'ga': 'en', 'gd': 'en', 'gv': 'en', 'is': 'da', 'kw': 'en',
                    'la': 'it', 'lb': 'de', 'nb': 'no', 'nn': 'na', 'oc': 'fr', 'wa': 'fr'}

    
def load_fasttext_model(language):
    # correctly load fasttext vectors
    if os.path.exists(f"data/fasttext/wiki.{language}.align.vec"):
        return KeyedVectors.load_word2vec_format(f"data/fasttext/wiki.{language}.align.vec")
    else:
        return None


def hf_dataset_to_arrays(dataset, embedding_fields, target_field='class'):
    """
    
    :param dataset:
    :param embedding_fields:
    :param target_field:
    :param class_encoder: 'onehot' or 'label' defines how to encode the class
    :return: X, y array of features and labels for XGBoost
    """
    # convert the dataset to numpy arrays
    X = np.concatenate([dataset[x] for x in embedding_fields], axis=1)
    
    
    for lang_field in ('lang1', 'lang2'):
        lang_index = np.array([LANGUAGES.index(x) for x in dataset[lang_field]])
        # encoding languages in one-hot representation
        one_hot_lf = np.zeros((len(lang_index), len(LANGUAGES)))
        one_hot_lf[np.arange(len(lang_index)), lang_index] = 1
        one_hot_lf = one_hot_lf[:, 1:]
        
        X = np.concatenate((X, one_hot_lf[:, 1:]), axis=1)

    if target_field is not None:
        y = np.array([CLASSES.index(x) for x in dataset[target_field]]).reshape(-1, 1)
    else:
        y = None
    
    return X, y


def train_xgboost(dataset_split, embedding_fields, target_field='class'):
    
    X, y = hf_dataset_to_arrays(dataset_split, embedding_fields, target_field)
    clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
                            gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
                            min_child_weight=1, missing=2, n_estimators=100, nthread=-1,
                            objective='multi:softprob', reg_alpha=0, reg_lambda=1,
                            scale_pos_weight=1, seed=0, silent=False, subsample=1)
    clf.fit(X, y)
    return clf


def infer_eval_xgboost(dataset_split, classifier, embedding_fields, target_field='class'):
    
    X, y = hf_dataset_to_arrays(dataset_split, embedding_fields, target_field)
    y_pred = classifier.predict(X)
    if target_field is not None:
        logging.info(f"Accuracy on the split: {accuracy_score(y, y_pred)}")
        
    y_pred_decoded = [CLASSES[x] for x in y_pred]
    return y_pred_decoded


def main():
    models = ['google/canine-c']  # ['fasttext', 'facebook-xlm-v-base']
    truncate_at = 500
    compr_c = 16
    dataset = ClassificationDataset(data_dir='data', truncate_at=truncate_at, dev_split=0.1)
    
    logging.info(f"Getting embeddings for train split from: {models}, dimensionality reduced to {compr_c}")
    compression_model = dataset.get_representation('train', ['word1', 'word2'], models, compress_components=compr_c)
    dataset.get_representation('validation', ['word1', 'word2'], models, compress_components=compr_c, compression_model=compression_model)
    dataset.get_representation('test', ['word1', 'word2'], models, compress_components=compr_c, compression_model=compression_model)
    
    embedding_fields = [f"{model}_{field}" for model in models for field in ['word1', 'word2']]
    
    logging.info(f"Training on {len(dataset.train)} samples")
    classifier = train_xgboost(dataset.train, embedding_fields)
    y_pred_train = infer_eval_xgboost(dataset.train, classifier, embedding_fields)
    logging.info(f"Confusion matrix, labels are: {CLASSES}, rows are true, columns are predicted")
    logging.info(f"\n{confusion_matrix(dataset.train['class'], y_pred_train, labels=CLASSES)}")
    
    logging.info(f"Evaluating on {len(dataset.validation)} samples:")
    y_pred_val = infer_eval_xgboost(dataset.validation, classifier, embedding_fields)
    logging.info(f"Confusion matrix:")
    logging.info(f"\n {confusion_matrix(dataset.validation['class'], y_pred_val, labels=CLASSES)}")
    
    logging.info(f"Predicting for evaluation split:")
    y_pred_test = infer_eval_xgboost(dataset.test, classifier, embedding_fields, target_field=None)
    
    # write the predictions to csv
    with open('data/eval_predictions.csv', 'w') as f:
        for i, pred in enumerate(y_pred_test):
            f.write(f"{i},{pred}\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

