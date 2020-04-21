
from pathlib import Path
import json

import lightgbm as lgb
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin

from sklearn.model_selection import train_test_split

from scipy.spatial.distance import cdist, pdist, squareform
import pandas as pd
import numpy as np


class BoosterEnsemble(object):

    def __init__(self, models):
        self.models = models

    def predict(self, x):
        result = np.zeros(x.shape[0])
        for m in self.models:
            result += m.predict(x,n_jobs=16)
        result /= len(self.models)
        return result

    def save_to_dir(self, directory):
        json_config_path = Path(directory) / 'ensemble.json'
        with json_config_path.open('w') as fout:
            json.dump(obj={'num_models': len(self.models)}, fp=fout)

        for i, m in enumerate(self.models):
            save_path = Path(directory) / f'booster{i}.bin'
            m.save_model(str(save_path))

    @classmethod
    def load_from_dir(cls, directory):
        json_config_path = Path(directory) / 'ensemble.json'
        with json_config_path.open('r') as fin:
            config = json.load(fin)

        return BoosterEnsemble(models=[
            lgb.Booster(model_file=str(Path(directory) / f'booster{i}.bin'))
            for i in range(config['num_models'])
        ])


class TableDist(object):

    def __init__(self, categoricals, numericals):
        self.cats = categoricals
        self.nums = numericals

        if len(self.cats) == 0 and len(self.nums) == 0:
            raise ValueError(f"either list of categoricals or list of numericals needs to be non-empty")
        # want the contribution of categorical features to be of same order
        # as the numerical features. numerical features are standardized
        # and the expected value of |x1 - x2| where x1, x2 ~ N(0, 1) is 2 / pi.
        # Since scipy computes normalized hamming distance we also have
        # to un-normalize using #categoricals
        # self.cat_scale = len(self.cats) * (2 / np.pi)

        self.cat_scale = len(self.cats)

    def __call__(self, a, b=None):
        # choosing l1 distance for numerical features as lightgbm will bin the
        # features. As such samples that are separated by a large l1 are more
        # prone to land in different leaf nodes of the tree.
        if b is None:
            # num_dists = squareform(pdist(a[self.nums], metric='cityblock'))
            if len(self.nums) > 0:
                num_dists = squareform(pdist(a[self.nums], metric='euclidean'))
            else:
                num_dists = 0.0

            if len(self.cats) > 0:
                cat_dists = squareform(pdist(a[self.cats], metric='hamming'))
            else:
                cat_dists = 0.0
        else:
            # num_dists = cdist(a[self.nums], b[self.nums], metric='cityblock')
            if len(self.nums) > 0:
                num_dists = cdist(a[self.nums], b[self.nums], metric='euclidean')
            else:
                num_dists = 0.0

            if len(self.cats) > 0:
                cat_dists = cdist(a[self.cats], b[self.cats], metric='hamming')
            else:
                cat_dists = 0.0

        dists = num_dists + self.cat_scale * cat_dists
        return dists


class TuningObjective(object):

    def __init__(self, train_set, valid_set, params, categorical_names):
        self.train = train_set
        self.valid = valid_set
        self.params = params
        self.catnames = categorical_names

    def __call__(self, hyperparams):
        model = lgb.train(
            {**self.params, **hyperparams},
            self.train,
            300,
            self.valid,
            categorical_feature=self.catnames,
            early_stopping_rounds=30,
            verbose_eval=0,
        )

        score = model.best_score["valid_0"][self.params["metric"]]

        return {'loss': -score, 'status': STATUS_OK}


def tune_hyper(x, y, params, seed, categorical_names, weights=None):
    if weights is None:
        weights = pd.Series(1.0, index=y.index)
    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(
        x,
        y,
        weights,
        test_size=0.5,
        random_state=seed,
        stratify=y,
        shuffle=True
    )
    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=categorical_names,
        free_raw_data=False,
        weight=w_train,
    )
    valid_data = lgb.Dataset(
        x_val,
        label=y_val,
        categorical_feature=categorical_names,
        free_raw_data=False,
        weight=w_val,
    )

    search_space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.8, 1.0, 0.1),
        "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }

    trials = Trials()
    tuning_obj = TuningObjective(
        train_set=train_data,
        valid_set=valid_data,
        params=params,
        categorical_names=categorical_names,
    )
    best = fmin(
        fn=tuning_obj,
        space=search_space,
        trials=trials,
        algo=tpe.suggest,
        max_evals=2,
        verbose=1,
        rstate=np.random.RandomState(seed),
    )

    hyperparams = space_eval(search_space, best)

    return hyperparams


def train_lgb(x, y, seed, categorical_names, weights=None):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": seed,
        "num_threads": 4
    }

    print(f'training with {x.shape[0]} samples')

    hyperparams = tune_hyper(
        x=x,
        y=y,
        params=params,
        seed=seed,
        categorical_names=categorical_names,
        weights=weights,
    )

    if weights is None:
        weights = pd.Series(1.0, index=y.index)

    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(
        x, y, weights,
        test_size=0.1,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    train = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=categorical_names,
        free_raw_data=False,
        weight=w_train,
    )
    valid = lgb.Dataset(
        x_val,
        label=y_val,
        categorical_feature=categorical_names,
        free_raw_data=False,
        weight=w_val,
    )
    model = lgb.train(
        {**params, **hyperparams},
        train,
        500,
        valid,
        categorical_feature=categorical_names,
        early_stopping_rounds=30,
        verbose_eval=100,
    )

    return model
