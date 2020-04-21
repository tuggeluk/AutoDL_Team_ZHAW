
import pandas as pd
import numpy as np

from trainer_pu import train_lgb, BoosterEnsemble
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
        self.models = []


    def train(self, x, y,num_models = 10,num_iter=2):
        y_scores = y
        max_size= int(len(y[(y == 1)])*1.5)
        num_pos = len(y[y==1])

        neg_thresholds = [0.5,0.5,0.5,0.5,0.5,0.5,0.5]
        pos_thresholds = [0.5,0.5,0.5,0.5,0.5,0.5,0.5]

        num_models = [5,5,10,10,10,10,10]

        for i in range(num_iter):
            print(i)
            y_scores_est = y_scores[(y_scores > pos_thresholds[i]) & (y==0)]
            y_pos_corr = y[(y == 1)]

            if len(y_scores_est) > 0:
                print(num_pos)
                pos = min(len(y_scores_est),num_pos)
                print(y_scores_est)
                print(pos)
                cutoff = np.partition(y_scores_est,-pos)[-pos]
            else:
                cutoff=1
            print("The cutoff is " + str(cutoff))


            for j in range(num_models[i]):

                y_pos_corr = y[(y == 1)]
                x_pos_corr = x.loc[y_pos_corr.index]
                y_pos_est = y[(y_scores > pos_thresholds[i]) & (y==0) & (y_scores>=cutoff)]


                print(len(y_pos_est))

                x_pos_est = x.loc[y_pos_est.index]
                if len(x_pos_est) > max_size - len(y_pos_corr):
                    x_pos_est = x_pos.sample(n=max_size - len(x_pos_corr)).reset_index(drop=True)
                x_pos = pd.concat([x_pos_corr,x_pos_est])
                y_pos = pd.Series(np.ones(len(x_pos), dtype=np.bool), index=x_pos.index)

                y_unk = y[(y_scores < neg_thresholds[i]) & (y!=1)]
                print(len(y_unk))
                x_unk = x.loc[y_unk.index]
                num_unk = min(len(x_unk),len(y_pos),max_size)
                x_unk = x_unk.sample(n=num_unk).reset_index(drop=True)

                x_neg = x_unk.sample(n=num_unk).reset_index(drop=True)
                y_neg = pd.Series(np.zeros(num_unk, dtype=np.bool), index=x_neg.index)

                x_lbl = pd.concat((x_pos, x_neg)).reset_index(drop=True)
                y_lbl = pd.concat((y_pos, y_neg)).reset_index(drop=True)
                self.models.append(train_lgb(x_lbl, y_lbl, self.seed, self.cats))
                final_model  = BoosterEnsemble(models=self.models)
            y_prev = y_scores
            y_scores = final_model.predict(x)
            self.models = []
            num_pos += len(y_pos_corr) + len(y_pos_est)
            print("num_pos is " + str(num_pos))

        return final_model
