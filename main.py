import os
import argparse
import random
import transformers
import json
import sys

from data_wrapper import ClassificationDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import class_weight

import xgboost as xgb
import numpy as np
from gensim.models import KeyedVectors

import logging

logging.basicConfig(level=logging.INFO)

LANGUAGES = ['en', 'fr', 'de', 'da', 'el', 'it', 'no', 'nl', 'af', 'br', 'ca', 'cs', 'cy', 'da', 'es',  'fy', 'ga', 'gd', 'gl', 'gv', 'is',
             'kw', 'la', 'lb', 'lt', 'lv', 'nb', 'nl', 'nn', 'no', 'oc', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv', 'wa']

CLASSES = ['der', 'cog', 'none']

FAMILIES = ['germanic', 'romance', 'hellenic', 'celtic', 'slavic', 'baltic']

# latin is not romance in theory but italic
language2family_mapping = {'en': 'germanic', 'fr': 'romance', 'de': 'germanic', 'da': 'germanic', 'el': 'hellenic',
                           'it': 'romance','no': 'germanic', 'nl': 'germanic', 'af': 'germanic', 'br': 'celtic',
                           'ca': 'romance', 'cs': 'slavic', 'cy': 'hellenic', 'es': 'romance', 'fy': 'germanic',
                           'ga': 'celtic', 'gd': 'celtic', 'gl': 'romance', 'gv': 'celtic', 'is': 'germanic',
                           'kw': 'celtic', 'la': 'romance', 'lb': 'germanic', 'lt': 'baltic', 'lv': 'baltic',
                           'nb': 'germanic', 'nn': 'germanic', 'oc': 'romance', 'pl': 'slavic', 'pt': 'romance',
                           'ro': 'romance', 'sk': 'slavic', 'sl': 'slavic', 'sv': 'germanic', 'wa': 'romance'}

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
        one_hot_lang = np.zeros((len(lang_index), len(LANGUAGES)))
        one_hot_lang[np.arange(len(lang_index)), lang_index] = 1
        # first column can be inferred from the other columns
        one_hot_lang = one_hot_lang[:, 1:]
        
        X = np.concatenate((X, one_hot_lang), axis=1)

        lang_family_index = np.array([FAMILIES.index(language2family_mapping[x]) for x in dataset[lang_field]])
        # encoding language families in one-hot representation
        one_hot_family = np.zeros((len(lang_family_index), len(FAMILIES)))
        one_hot_family[np.arange(len(lang_family_index)), lang_family_index] = 1
        one_hot_family = one_hot_family[:, 1:]
        
        X = np.concatenate((X, one_hot_family), axis=1)

    if target_field is not None:
        y = np.array([CLASSES.index(x) for x in dataset[target_field]]).reshape(-1, 1)
    else:
        y = None
    
    return X, y


def train_xgboost(dataset_split, embedding_fields, target_field='class', weighting=True, seed=1234):
    
    X, y = hf_dataset_to_arrays(dataset_split, embedding_fields, target_field)
    
    xgboost_params = {
        'objective': 'multi:softprob',
        'missing': 2,
        'base_score': 0.5,
        'colsample_bylevel': 1,
        'colsample_bytree': 1,
        'gamma': 0,
        'learning_rate': 0.1,
        'max_delta_step': 1,
        'max_depth': 6,
        'min_child_weight': 1,
        'n_estimators': 100,
        'nthread': -1,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'scale_pos_weight': 1,
        'subsample': 1,
        'verbosity': 2,
        'seed': seed}
    
    if weighting:
        sample_weights = class_weight.compute_sample_weight('balanced', y)
        xgboost_params['sample_weight'] = sample_weights
        logging.info(f"Using sample weights: {sample_weights}")
    
    clf = xgb.XGBClassifier(**xgboost_params)

    clf.fit(X, y)

    return clf


def infer_eval_xgboost(dataset_split, classifier, embedding_fields, target_field='class'):
    
    X, y = hf_dataset_to_arrays(dataset_split, embedding_fields, target_field)
    y_pred_probs = classifier.predict_proba(X)
    y_pred = np.argmax(y_pred_probs, axis=1)
    if target_field is not None:
        acc = accuracy_score(y, y_pred)
        logging.info(f"Accuracy on the split: {acc}")
    else:
        acc = None
    
    y_pred_decoded = [CLASSES[x] for x in y_pred]
    return y_pred_decoded, y_pred_probs, acc


def save_predictions_and_model(classifier, predictions, probabilities, results, output_dir, models, truncate_at, compress_components, weighting, seed):
    
    output_path = os.path.join(output_dir, f"models_{'_'.join(models).replace('/','-')}_cc_{compress_components}_seed_{seed}")
    if truncate_at > 0:
        output_path+= f"_truncate_{truncate_at}"
    output_path += "_family"
    if weighting:
        output_path += "_weighted"

    output_directory = os.path.join(output_dir, output_path)
    os.makedirs(output_directory, exist_ok=True)

    # write the predictions to csv
    with open(os.path.join(output_directory, 'eval_predictions.csv'), 'w') as f:
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")
            
    with open(os.path.join(output_directory, 'eval_predictions+probabilities.csv'), 'w') as f:
        for i, (pred, probs) in enumerate(zip(predictions,probabilities)):
            f.write(f"{i},{pred},{probs}\n")
    # save the model
    classifier.save_model(os.path.join(output_directory, 'model.json'))
    
    # save the results
    with open(os.path.join(output_directory, 'results.json'), 'w') as f:
        json.dump(results, f)
        
    # save the code
    with open(sys.argv[0], 'r') as cur_file:
        cur_running = cur_file.readlines()
    with open(os.path.join(output_directory,'script.py'),'w') as log_file:
        log_file.writelines(cur_running)
    
    
def run_xgboost(dataset, embedding_fields, weighting, seed):
    results = {'train': {}, 'validation': {}}  # store the results for each split
    
    logging.info(f"Training on {len(dataset.train)} samples")
    classifier = train_xgboost(dataset.train, embedding_fields, weighting=weighting, seed=seed)
    y_pred_train, _, acc_train = infer_eval_xgboost(dataset.train, classifier, embedding_fields)
    logging.info(f"Confusion matrix, labels are: {CLASSES}, rows are true, columns are predicted")
    train_cm = confusion_matrix(dataset.train['class'], y_pred_train, labels=CLASSES)
    logging.info(f"\n{train_cm}")
    
    results['train']['acc'] = acc_train
    results['train']['cm'] = train_cm.tolist()
    
    logging.info(f"Evaluating on {len(dataset.validation)} samples:")
    y_pred_val, _, acc_val = infer_eval_xgboost(dataset.validation, classifier, embedding_fields)
    logging.info(f"Confusion matrix:")
    val_cm = confusion_matrix(dataset.validation['class'], y_pred_val, labels=CLASSES)
    logging.info(f"\n {val_cm}")
    results['validation']['acc'] = acc_val
    results['validation']['cm'] = val_cm.tolist()
    
    logging.info(f"Predicting for evaluation split:")
    y_pred_test, y_pred_probs_test, _ = infer_eval_xgboost(dataset.test, classifier, embedding_fields, target_field=None)
    
    return y_pred_test, y_pred_probs_test, classifier, results


def main(models, truncate_at, compress_components, data_dir, output_dir, seed):
    # models = ['google/canine-c']  # ['fasttext', 'facebook-xlm-v-base']
    # truncate_at = 500
    # compress_components = 16
    weighting = True
    hf_cache = f"cached_models_{'_'.join(models).replace('/','-')}"
    dataset = ClassificationDataset(data_dir=data_dir, hf_cache=hf_cache, truncate_at=truncate_at, dev_split=0.1, seed=seed)
    
    if not dataset.loaded_from_cache:
        logging.info(f"Getting embeddings for train split from: {models}, dimensionality reduced to {compress_components}")
        compression_models = dataset.get_representation('train', ['word1', 'word2'], models, compress_components=compress_components)
        dataset.get_representation('validation', ['word1', 'word2'], models, compress_components=compress_components, compression_models=compression_models)
        dataset.get_representation('test', ['word1', 'word2'], models, compress_components=compress_components, compression_models=compression_models)
        dataset.save_hf_cache(os.path.join(data_dir, hf_cache))
    else:
        logging.info(f"Loaded dataset from cache: {hf_cache}")
        
    embedding_fields = [f"{model}_{field}" for model in models for field in ['word1', 'word2']]
    test_prediction, test_probs, classifier, results = run_xgboost(dataset, embedding_fields, weighting, seed)
    save_predictions_and_model(classifier, test_prediction, test_probs, results, output_dir,
                               models, truncate_at, compress_components, weighting,  seed)
    
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--truncate_at', type=int, default=500)
    parser.add_argument('--compress_components', type=int, default=16)
    parser.add_argument('--models', nargs='+', default=['google/canine-c'])
    parser.add_argument('--seed', type=int, default=1234)
    
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.models, args.truncate_at, args.compress_components, args.data_dir, args.output_dir, args.seed)
    
