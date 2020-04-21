
import pandas as pd
import numpy as np

from trainer import train_lgb, BoosterEnsemble
import time

class SemiSupervised(object):

    def __init__(self, seed, categorical_names, noisy_trainer,time_budget):
        self.cats = categorical_names
        self.seed = seed
        self.noisy = noisy_trainer
        self.time_budget = time_budget
    def train(self, x, y):
        y_lbl = pd.concat([y[y == -1], y[y == 1]]) > 0
        frac_pos = np.sum(y_lbl) / len(y_lbl)
        x_lbl = x.loc[y_lbl.index]
        y_unk = y[y == 0]  # not used after this
        x_unk = x.loc[y_unk.index]

        ensemble = BoosterEnsemble(models=[
            train_lgb(
                x=x_lbl,
                y=y_lbl,
                seed=self.seed,
                categorical_names=self.cats,
            )
        ])

        current_index = 0
        step = 5000
        end_index = min(x_lbl.shape[0], 10000)
        while current_index < end_index:
            start = time.time()
            print(current_index, end_index)

            x_new = x_unk.iloc[current_index:min(current_index+step,end_index)]


            y_new = ensemble.predict(x_new)
            y_new = y_new > np.quantile(y_new, q=1 - frac_pos)
            y_new = pd.Series(y_new, index=x_new.index)

            noisy_model = self.noisy.train(x_new, y_new,n_models=1)
            end = time.time()
            print(current_index)

            if current_index==0:
                time_avg = end - start

            ensemble.models += noisy_model.models

            self.time_budget -= end -start
            current_index += step

            if self.time_budget < 1.2*time_avg:
                break

        return ensemble


class OldSemiSupervised(object):

    def __init__(self, seed, categorical_names):
        self.cats = categorical_names
        self.seed = seed

    def train(self, x, y):
        y_lbl = pd.concat([y[y == -1], y[y == 1]]) > 0
        frac_pos = np.sum(y_lbl) / len(y_lbl)
        x_lbl = x.loc[y_lbl.index]
        y_unk = y[y == 0]  # not used after this
        x_unk = x.loc[y_unk.index]

        data_split = {}
        for i in range(3):
            y = y_lbl.sample(frac=0.5)
            x = x_lbl.loc[y.index]
            data_split[i] = {'x': x, 'y': y}

        models = {
            i: train_lgb(
                **data,
                seed=self.seed,
                categorical_names=self.cats,
                weights=None
            )
            for i, data in data_split.items()
        }

        current_index = 0
        step = 10000
        end_index = min(x_lbl.shape[0], 50000)
        while current_index < end_index:
            print(current_index, end_index)
            x_new = x_unk.iloc[current_index:current_index+step]
            current_index += step

            preds = {
                i: model.predict(x_new)
                for i, model in models.items()
            }

            preds = {
                i: probas > np.quantile(probas, q=1 - frac_pos)
                for i, probas in preds.items()
            }

            n_skipped = 0
            for i, ix in enumerate(x_new.index):
                p0 = preds[0][i]
                p1 = preds[1][i]
                p2 = preds[2][i]
                agree_01 = p0 == p1
                agree_02 = p0 == p2
                agree_12 = p1 == p2
                all_agree = p0 == p1 == p2

                if all_agree:
                    n_skipped += 1
                    continue
                elif agree_01:
                    data_split[2]['x'] = data_split[2]['x'].append(x_new.loc[ix])
                    data_split[2]['y'] = data_split[2]['y'].append(pd.Series([not p2], index=[ix]))
                elif agree_02:
                    data_split[1]['x'] = data_split[1]['x'].append(x_new.loc[ix])
                    data_split[1]['y'] = data_split[1]['y'].append(pd.Series([not p1], index=[ix]))
                elif agree_12:
                    data_split[0]['x'] = data_split[0]['x'].append(x_new.loc[ix])
                    data_split[0]['y'] = data_split[0]['y'].append(pd.Series([not p0], index=[ix]))
                else:
                    # this shouldn't happen
                    raise ValueError(f"big bug in tri-learning")

            print('skipped: ', n_skipped)

            models = {
                i: train_lgb(
                    **data,
                    seed=self.seed,
                    categorical_names=self.cats,
                    weights=None
                )
                for i, data in data_split.items()
            }

        return BoosterEnsemble(
            models=list(models.values())
        )
