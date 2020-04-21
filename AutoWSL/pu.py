
import pandas as pd
import numpy as np

from trainer import train_lgb, BoosterEnsemble
from noisy import PrototypeMiner


class PUTrainer(object):

    def __init__(self, seed, categorical_names, numerical_names):
        self.seed = seed
        self.cats = categorical_names
        self.nums = numerical_names
        self.proto_miner = PrototypeMiner(
            categoricals=self.cats,
            numericals=self.nums,
        )

    def train(self, x, y):
        y_pos = y[y == 1]
        x_pos = x.loc[y_pos.index]
        y_unk = y[y == 0]
        x_unk = x.loc[y_unk.index]
        x_unk = x_unk.sample(n=10000).reset_index(drop=True)

        # pos_protos = self.proto_miner.mine(
        #     x=x_pos,
        #     n_sample=1000,
        #     n_protoypes=1000,
        # )
        # pos_dists = pd.Series(
        #     self.proto_miner.dist_fn(x_unk, pos_protos).mean(axis=1),
        #     index=x_unk.index
        # )

        # neg_index = pos_dists.argsort()[-len(y_pos):]
        # x_neg = x_unk.iloc[neg_index]
        x_neg = x_unk.sample(n=len(y_pos)).reset_index(drop=True)
        y_neg = pd.Series(np.zeros(len(y_pos), dtype=np.bool), index=x_neg.index)

        x_lbl = pd.concat((x_pos, x_neg)).reset_index(drop=True)
        y_lbl = pd.concat((y_pos, y_neg)).reset_index(drop=True)

        return BoosterEnsemble(
            models=[train_lgb(x_lbl, y_lbl, self.seed, self.cats)]
        )
