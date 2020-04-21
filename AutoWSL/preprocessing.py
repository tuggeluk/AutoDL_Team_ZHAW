
import pickle
import json
from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler


class Preprocessor(object):

    def __init__(self, schema):
        self.schema = schema
        self.category_maps = {}
        self.to_drop = set()
        self.to_norm = []
        self.scaler = StandardScaler()

    def fit(self, x: pd.DataFrame):
        self.category_maps = {
            column_name: {
                value: i
                for i, value in enumerate(x[column_name].unique())
            }
            for column_name in x.columns
            if self.schema.get(column_name) == 'cat'
        }
        self.to_drop = [
            column_name
            for column_name in x.columns
            if not (self.schema.get(column_name) == 'cat' or self.schema.get(column_name) == 'num')
        ]
        #print(self.category_maps.keys())
        for column_name in x.columns:
            if self.schema.get(column_name) == 'cat':
                if len(x[column_name].unique()) > 500:
                    self.to_drop.append(column_name)
                    del self.category_maps[column_name]
        #print(self.category_maps.keys())
        #print(self.to_drop)
        self.to_norm = [
            column_name
            for column_name in x.columns
            if self.schema.get(column_name) == 'num'
        ]
        #print(self.to_norm)
        x_copy = x.drop(columns=self.to_drop)
        self.scaler.fit(x_copy[self.to_norm])

        return self

    def transform(self, x: pd.DataFrame):
        x.drop(columns=self.to_drop, inplace=True)
        for col_name in self.category_maps.keys():

            relevant_map = self.category_maps[col_name]
            x[col_name] = getattr(x, col_name).apply(
                lambda entry: relevant_map.get(entry, len(relevant_map))
            )

        x[self.to_norm] = self.scaler.transform(x[self.to_norm])
        #print(x.head(5))
        return x

    def fit_transform(self, x: pd.DataFrame):
        return self.fit(x).transform(x)

    def save(self, directory: str):
        save_path = Path(directory) / 'preprocessor.json'

        state = {
            'schema': self.schema,
            'category_maps': self.category_maps,
            'to_drop': list(self.to_drop),
            'to_norm': self.to_norm,
        }

        with save_path.open('w') as fout:
            json.dump(fp=fout, obj=state)

        scaler_path = Path(directory) / 'scaler.pkl'
        with scaler_path.open('wb') as fout:
            pickle.dump(obj=self.scaler, file=fout)

    def load(self, directory: str):
        load_path = Path(directory) / 'preprocessor.json'

        with load_path.open('r') as fin:
            state = json.load(fin)

        self.schema = state['schema']
        self.category_maps = state['category_maps']
        self.to_drop = state['to_drop']
        self.to_norm = state['to_norm']

        scaler_path = Path(directory) / 'scaler.pkl'
        with scaler_path.open('rb') as fin:
            self.scaler = pickle.load(fin)
