import os
import argparse
import random
import transformers
import json
import sys
from functools import partial

from data_wrapper import ClassificationDataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

import xgboost as xgb
import numpy as np
from gensim.models import KeyedVectors
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

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


def objective(space, X_train, X_test, y_train, y_test, sample_weights=None):
    space['max_depth'] = int(space['max_depth'])
    space['min_child_weight'] = int(space['min_child_weight'])
    
    clf = xgb.XGBClassifier(**space)
    
    evaluation = [(X_train, y_train), (X_test, y_test)]
    clf.fit(X_train, y_train, eval_set=evaluation, sample_weight=sample_weights)
    pred = clf.predict(X_test)
    f1 = f1_score(y_test, pred, average='macro')
    return {'loss': 1 - f1, 'status': STATUS_OK}


def hyperparam_optimization(X, y,weighting, **params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    if weighting:
        sample_weights = class_weight.compute_sample_weight('balanced', y_train)

    else:
        sample_weights = None
    space = {
        'max_depth': hp.quniform('max_depth', 3, 20, 1),
        'min_child_weight':  hp.quniform('min_child_weight', 1, 6, 1),
        'gamma': hp.uniform('gamma', 0, 5),
        'eta': hp.uniform('eta', 0.01, 0.3),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.6, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
        'subsample': hp.uniform('subsample', 0.6, 1)
    }
    
    params = {param: params[param] for param in params if param in space}
    params.update(space)

    trials = Trials()
    fmin_objective = partial(objective, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                             sample_weights=sample_weights)
    best = fmin(fn=fmin_objective,
                space=params,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)
    
    logging.info(f"Best hyperparameters: {best}")
    params.update(best)
    return best
    

def train_xgboost(dataset_split, embedding_fields, target_field='class', weighting=True, hyperparam_optimize = True, seed=1234):
    
    X, y = hf_dataset_to_arrays(dataset_split, embedding_fields, target_field)
    
    general_params = {
        'booster': 'gbtree',
        'verbosity': 1,
        'nthread': -1,
        'early_stopping_rounds': 10
    }

    # to tune: max_depth 3 - 10, min_child_weight 1 - 6, gamma 0 - 0.3, eta 0.01 - 0.3, colsample_bylevel 0.6 - 1
    
    booster_params = {
        'eta': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'max_delta_step': 5,
        'subsample': 0.7,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 0.7,
        'colsample_bynode': 1.0,
        'lambda': 1.0,
        'alpha': 0.0,
    
    }
    
    task_params = {
        'objective': 'multi:softmax',
        'num_class': len(CLASSES),
        'eval_metric': 'aucpr',
        'seed': seed
    }
    

        
    if hyperparam_optimize:
        final_params = hyperparam_optimization(X, y, weighting, **general_params, **booster_params, **task_params)
        final_params['max_depth'] = int(final_params['max_depth'])
        final_params['min_child_weight'] = int(final_params['min_child_weight'])
    else:
        final_params = {**general_params, **booster_params, **task_params}
    clf = xgb.XGBClassifier(**final_params)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    evaluation = [(X_train, y_train), (X_test, y_test)]
    
    if weighting:
        sample_weights = class_weight.compute_sample_weight('balanced', y_train)
        logging.info(f"Using sample weights: {sample_weights}")
    else:
        sample_weights = None
        
    clf.fit(X_train, y_train, eval_set=evaluation, sample_weight=sample_weights)

    return clf


def infer_eval_xgboost(dataset_split, classifier, embedding_fields, target_field='class'):
    
    X, y = hf_dataset_to_arrays(dataset_split, embedding_fields, target_field)
    y_pred_probs = classifier.predict_proba(X)
    y_pred = np.argmax(y_pred_probs, axis=1)
    if target_field is not None:
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')
        logging.info(f"Accuracy on the split: {acc}")
        logging.info(f"F1 on the split: {f1}")
    else:
        acc = None
        f1 = None
    
    y_pred_decoded = [CLASSES[x] for x in y_pred]
    return y_pred_decoded, y_pred_probs, acc, f1


def save_predictions_and_model(classifier,results, validation_tes, test_tes, output_dir, models, truncate_at,
                               compress_components, weighting, hyperparam_optimize, seed):
    
    output_path = os.path.join(output_dir, f"models_{'_'.join(models).replace('/','-')}_cc_{compress_components}_seed_{seed}")
    if truncate_at > 0:
        output_path+= f"_truncate_{truncate_at}"
    output_path += "_family"
    if weighting:
        output_path += "_weighted"
    if hyperparam_optimize:
        output_path += "_hyperopt"

    output_directory = os.path.join(output_dir, output_path)
    os.makedirs(output_directory, exist_ok=True)

    with open(os.path.join(output_directory, 'validation_predictions.txt'), 'w') as f:
        for i, (eval_line, pred) in enumerate(zip(validation_tes, results['validation']['predictions'])):
            f.write(f"{eval_line},{pred}\n")
    
    # write the predictions to csv
    with open(os.path.join(output_directory, 'eval_predictions.csv'), 'w') as f:
        for i, (eval_line, pred) in enumerate(zip(test_tes, results['test']['predictions'])):
            f.write(f"{eval_line},{pred}\n")
            
    with open(os.path.join(output_directory, 'eval_predictions+probabilities.csv'), 'w') as f:
        for i, (eval_line,pred, probs) in enumerate(zip(test_tes, results['test']['predictions'], results['test']['probabilities'])):
            f.write(f"{eval_line},{pred},{probs}\n")
    # save the model
    classifier.save_model(os.path.join(output_directory, 'model.json'))
    
    results_pruned = {k: {k2: v2 for k2, v2 in v.items() if k2 in {'cm', 'acc'}} for k, v in results.items()}
    # save the results
    with open(os.path.join(output_directory, 'results.json'), 'w') as f:
        json.dump(results_pruned, f)
        
    # save the code
    with open(sys.argv[0], 'r') as cur_file:
        cur_running = cur_file.readlines()
    with open(os.path.join(output_directory,'script.py'),'w') as log_file:
        log_file.writelines(cur_running)
    
    
def run_xgboost(dataset, embedding_fields, weighting, hyperparam_optimize, seed):
    
    results = {'train': {}, 'validation': {}, 'test': {}}  # store the results for each split
    
    logging.info(f"Training on {len(dataset.train)} samples")
    classifier = train_xgboost(dataset.train, embedding_fields,
                               weighting=weighting, hyperparam_optimize=hyperparam_optimize, seed=seed)
    y_pred_train, y_pred_probs_train, acc_train, f1_train = infer_eval_xgboost(dataset.train, classifier, embedding_fields)
    logging.info(f"Confusion matrix, labels are: {CLASSES}, rows are true, columns are predicted")
    train_cm = confusion_matrix(dataset.train['class'], y_pred_train, labels=CLASSES)
    logging.info(f"\n{train_cm}")
    
    results['train']['acc'] = acc_train
    results['train']['f1'] = f1_train
    results['train']['cm'] = train_cm.tolist()
    results['train']['predictions'] = y_pred_train
    results['train']['probabilities'] = y_pred_probs_train
    
    logging.info(f"Evaluating on {len(dataset.validation)} samples:")
    y_pred_val, y_pred_probs_val, acc_val, f1_val = infer_eval_xgboost(dataset.validation, classifier, embedding_fields)
    logging.info(f"Confusion matrix:")
    val_cm = confusion_matrix(dataset.validation['class'], y_pred_val, labels=CLASSES)
    logging.info(f"\n {val_cm}")
    results['validation']['acc'] = acc_val
    results['validation']['f1'] = f1_val
    results['validation']['cm'] = val_cm.tolist()
    results['validation']['predictions'] = y_pred_val
    results['validation']['probabilities'] = y_pred_val
    
    logging.info(f"Predicting for evaluation split:")
    y_pred_test, y_pred_probs_test, _, _ = infer_eval_xgboost(dataset.test, classifier, embedding_fields, target_field=None)
    results['test']['predictions'] = y_pred_test
    results['test']['probabilities'] = y_pred_probs_test
    
    return classifier, results


def main(models, truncate_at, compress_components, data_dir, output_dir, seed):
    # models = ['google/canine-c']  # ['fasttext', 'facebook-xlm-v-base']
    # truncate_at = 500
    # compress_components = 16
    weighting = True
    hyperparam_optimize = True
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
    validation_tes = [','.join([te['word1'], te['lang1'], te['word2'], te['lang2'], te['class']]) for te in dataset.validation]
    test_tes = [','.join([te['word1'], te['lang1'], te['word2'], te['lang2']]) for te in dataset.test]
    
    classifier, results = run_xgboost(dataset, embedding_fields, weighting, hyperparam_optimize, seed)
    save_predictions_and_model(classifier, results, validation_tes, test_tes, output_dir,
                               models, truncate_at, compress_components, weighting, hyperparam_optimize,  seed)
    
    
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
    
