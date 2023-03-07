import os
import random

import numpy as np
import pandas as pd
import transformers
from sklearn.decomposition import TruncatedSVD
from datasets import Dataset


class ClassificationDataset:
    
    NUM_CLASSES = 3
    random.seed(42)
    
    def __init__(self, data_dir, hf_cache=None,  truncate_at=-1, dev_split=0., seed=1234):
        
        self.seed = seed
        
        self.data_dir = data_dir
        self.dev_split = dev_split
        
        self.truncate_at = truncate_at
        self.dataset = {}
        
        self.loaded_from_cache = False
        
        if hf_cache is not None and os.path.exists(os.path.join(data_dir, hf_cache)):
            self.load_hf_cache(os.path.join(data_dir, hf_cache))
            self.loaded_from_cache = True
        else:
            self.load_csv()
            
    def load_csv(self):
        
        train_data = pd.read_csv(os.path.join(self.data_dir, 'train.csv'),
                                 names=['word1', 'lang1', 'word2', 'lang2', 'class'], na_filter=False)
        train_data = Dataset.from_pandas(train_data)
        
        if self.truncate_at > 0:
            train_data = train_data.shuffle(seed=self.seed).select(range(self.truncate_at))
            
        if self.dev_split > 0:
            # split dataset into train and dev
            self.dataset = train_data.train_test_split(test_size=self.dev_split)
            self.dataset['validation'] = self.dataset['test']
        else:
            self.dataset['train'] = train_data
            self.dataset['validation'] = None

        test_data = pd.read_csv(os.path.join(self.data_dir, 'eval.csv'), names=['word1', 'lang1', 'word2', 'lang2'],
                                na_filter=False)
        
        test_data = Dataset.from_pandas(test_data)
        test_data = test_data.add_column('class', ['none'] * len(test_data))
        
        self.dataset['test'] = test_data

    def load_hf_cache(self, hf_cache_path):
        for dataset_split in ['train', 'validation', 'test']:
            self.dataset[dataset_split] = Dataset.load_from_disk(os.path.join(hf_cache_path, dataset_split))
        
    def save_hf_cache(self, hf_cache_path):
        for dataset_split in ['train', 'validation', 'test']:
            self.dataset[dataset_split].save_to_disk(os.path.join(hf_cache_path, dataset_split))
        
    @staticmethod
    def compress_representation(in_representation, compress_components=8, compression_model=None):

        in_representation = np.array([x for x in in_representation])
        if compression_model is None:
            # train SVD to compress the representations
            compression_model = TruncatedSVD(n_components=compress_components, random_state=0)
            compression_model.fit(in_representation)
            out_representation = compression_model.transform(in_representation)

        else:
            # use existing SVD to compress the representations
            out_representation = compression_model.transform(in_representation)

        out_representation = [list(row) for row in out_representation]
        return out_representation, compression_model

    def get_representation(self, dataset_split, fields, models_names, compress_components=None, compression_models=None):
        """
        :param dataset: dataset to get the representation from
        :param fields: fields of dataset to get the representation
        :param models_names: embedding models to use for the representation
        """
    
        # if 'fasttext' in models_names:
        #     ft_models = {}
        #     for lang in LANGUAGES:
        #         model = load_fasttext_model(lang)
        #         if model is None:
        #             ft_models[lang] = ft_models[fasttext_mapping.get(lang, 'en')]
        #         else:
        #             ft_models[lang] = model
        
        if compression_models is None:
            compression_models = {models_name : None for models_name in models_names}
    
        for idx, field in enumerate(fields):
            for model_name in models_names:
            
                # if model_name == 'fasttext':
                #     if field == 'word1':
                #         dataset = dataset.map(lambda x: {f"{model_name}_{field}": ft_models[x['lang1']].wv[x[field]]},
                #                               batched=True)
                #     else:
                #         dataset = dataset.map(lambda x: {f"{model_name}_{field}": ft_models[x['lang2']].wv[x[field]]},
                #                               batched=True)
                # else:
                model = transformers.AutoModel.from_pretrained(model_name, max_length=16)
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, model_max_length=16)

                self.dataset[dataset_split] = self.dataset[dataset_split].map(lambda x: {f"{model_name}_{field}": model(
                    **tokenizer(x[field],
                                return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
                )[0].mean(1)}, batched=True)

                if compression_models[model_name] is not None:
                    compressed_representation, _ = self.compress_representation(self.dataset[dataset_split][f"{model_name}_{field}"],
                                                                                compress_components=compress_components,
                                                                                compression_model=compression_models[model_name])

                elif compress_components is not None:
                    compressed_representation, compression_models[model_name]= self.compress_representation(
                        self.dataset[dataset_split][f"{model_name}_{field}"], compress_components=compress_components)

                else:
                    continue
            
                # update the embeddings with compressed values
                self.dataset[dataset_split]  = self.dataset[dataset_split].remove_columns(
                    [f"{model_name}_{field}"]).add_column(f"{model_name}_{field}", compressed_representation)
    
        return compression_models

    @property
    def train(self):
        return self.dataset['train']
    
    @property
    def validation(self):
        return self.dataset['validation']
    
    @property
    def test(self):
        return self.dataset['test']
    
    #setters
    @train.setter
    def train(self, value):
        self.dataset['train'] = value
        
    @validation.setter
    def validation(self, value):
        self.dataset['validation'] = value
        
    @test.setter
    def test(self, value):
        self.dataset['test'] = value


