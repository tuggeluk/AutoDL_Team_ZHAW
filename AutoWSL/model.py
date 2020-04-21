
import pandas as pd

from trainer import BoosterEnsemble
from preprocessing import Preprocessor

from noisy import Noisy
from semi import SemiSupervised
from pu_modified import PUTrainer


class Model:

    def __init__(self, info: dict):
        self.info = info
        self.preprocessor = Preprocessor(schema=info['schema'])
        self.seed = 0xDEADBEEF
        self.model = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        x = self.preprocessor.fit_transform(X)

        if self.info['task'] == 'noisy':
            trainer = Noisy(
                seed=self.seed,
                categorical_names=list(self.preprocessor.category_maps.keys()),
                numerical_names=self.preprocessor.to_norm,
                time_budget = self.info["time_budget"],
                alpha=0.4,
            )
        elif self.info['task'] == 'ssl':
            trainer = SemiSupervised(
                seed=self.seed,
                categorical_names=list(self.preprocessor.category_maps.keys()),
                time_budget = self.info["time_budget"],
                noisy_trainer=Noisy(
                    seed=self.seed,
                    categorical_names=list(self.preprocessor.category_maps.keys()),
                    numerical_names=self.preprocessor.to_norm,
                    alpha=0.4,
                    time_budget = self.info["time_budget"],
                )
            )
        elif self.info['task'] == 'pu':
            trainer = PUTrainer(
                seed=self.seed,
                categorical_names=list(self.preprocessor.category_maps.keys()),
                numerical_names=self.preprocessor.to_norm,
            )
        else:
            raise ValueError(f"unsupported task {self.info['task']}")

        self.model = trainer.train(x, y)

    def predict(self, X: pd.DataFrame):
        if len(X) < 10000:
            x = self.preprocessor.transform(X)

            return pd.Series(self.model.predict(x))
        else:
            current_index = 0
            step_size=40000
            end_index = len(X)
            predictions = []
            while current_index < end_index:
                print(current_index)
                x_small = X.iloc[current_index:min(end_index,current_index+step_size)]

                x_small = self.preprocessor.transform(x_small)
                if current_index == 0 :
                    predictions = list(self.model.predict(x_small))
                else:
                    predictions += list(self.model.predict(x_small))
                print("The length is ", len(predictions))
                current_index += step_size

            return pd.Series(predictions)


    def save(self, directory: str):
        self.preprocessor.save(directory)
        self.model.save_to_dir(directory)

    def load(self, directory: str):
        self.preprocessor.load(directory)
        self.model = BoosterEnsemble.load_from_dir(directory)
