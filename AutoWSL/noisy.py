import time
import pandas as pd
import numpy as np

from trainer import train_lgb, TableDist, BoosterEnsemble


class PrototypeMiner(object):

    def __init__(self, categoricals, numericals):
        self.dist_fn = TableDist(
            categoricals=categoricals,
            numericals=numericals,
        )

    def mine(self, x, n_sample=1000, n_protoypes=100, cutoff_type=None, cutoff=0.95):
        """
        :param x: dataframe from which to mine prototypes
        :param n_sample: number of samples to consider
        :param n_protoypes: number of protoype rows to return
        :param cutoff_type: None or 'no' for not using nu values from paper
        'abs' or 'absolute' for using nu as in paper with an absolute cutoff
        'rel', 'relative', 'quant', 'quantile' for cutoff according to quantile
        :param cutoff:
        :return:
        """
        sample_data = x.sample(n=min(n_sample, x.shape[0]))

        dists = pd.DataFrame(
            self.dist_fn(sample_data),
            columns=sample_data.index,
            index=sample_data.index,
        )
        sims = 1.0 / (1.0 + dists)

        s_c = np.quantile(sims.values, q=0.6)  # as defined in the paper
        rhos = np.sign(sims - s_c).sum(axis=1)
        rho_max = np.max(rhos.values)

        nu = pd.Series(index=rhos.index)
        for ix in nu.index:
            rho_i = rhos.loc[ix]
            if rho_i < rho_max:
                nu.loc[ix] = max(sims.loc[ix, jx] for jx in rhos.index if rhos.loc[jx] > rho_i)
            else:
                nu.loc[ix] = min(sims.loc[ix])

        if cutoff_type is None or cutoff_type.startswith('no'):
            nu_filter = pd.Series(True, index=rhos.index)
        elif cutoff_type.startswith('abs'):
            nu_filter = nu < cutoff
        elif cutoff_type.startswith('rel') or cutoff_type.startswith('quant'):
            nu_filter = nu < np.quantile(nu, q=cutoff)
        else:
            raise ValueError(f"unknown cutoff type '{cutoff_type}'")

        prototype_index = rhos[nu_filter].sort_values(ascending=False).head(n_protoypes).index

        return x.loc[prototype_index]

    def corrected_labels(self, x, y):
        print('mine pos protos')
        pos_protos = self.mine(
            x.loc[y > 0],
        )

        print('mine neg protos')
        neg_protos = self.mine(
            x.loc[y <= 0],
        )

        print('compute pos dist')
        pos_dists = pd.Series(self.dist_fn(x, pos_protos).mean(axis=1), index=x.index)
        print('compute neg dist')
        neg_dists = pd.Series(self.dist_fn(x, neg_protos).mean(axis=1), index=x.index)

        y_corr = pos_dists < neg_dists
        return y_corr


class Noisy(object):

    def __init__(self, alpha, seed, categorical_names, numerical_names,time_budget):
        self.alpha = alpha
        self.seed = seed
        self.cats = categorical_names
        self.nums = numerical_names
        self.miner = PrototypeMiner(categoricals=self.cats, numericals=self.nums)
        self.time_budget = time_budget

    def train(self, x, y,n_models=20):
        y_obs = y > 0
        y_corr = self.miner.corrected_labels(x, y)
        w_obs = pd.Series(1 - self.alpha, index=y_obs.index)
        w_corr = pd.Series(self.alpha, index=y_corr.index)
        x = pd.concat((x, x))
        y = pd.concat((y_obs, y_corr))
        w = pd.concat((w_obs, w_corr))
        models = []
        for i in range(n_models):
            start = time.time()
            model = train_lgb(x, y, self.seed+i, self.cats, weights=w)
            models.append(model)
            end = time.time()
            if i==0:
                time_avg = end - start

            self.time_budget -= end -start
            if self.time_budget < 1.2*time_avg:
                break

        return BoosterEnsemble(models=models)
