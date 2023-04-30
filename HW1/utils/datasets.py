import numpy as np
from abc import ABC, abstractmethod
from utils.shuffle import unison_shuffled_copies
import pandas as pd


from sklearn.model_selection import train_test_split



class BaseSplitter(ABC):

    def __init__(self, split_size: int = 0.8):
        self.split_size = split_size


    @abstractmethod
    def split(self, X, Y):
        ...



class CustomSplitter(BaseSplitter):

    def __init__(self, monotonic_x = True):
        super().__init__()
        self.monotonic_x = monotonic_x

    def split(self, X, Y):
        X, Y = unison_shuffled_copies(X, Y)
        idx = int(self.split_size * len(X))
        dataset = {el:{set_type:None for set_type in ('train', 'test')} for el in ('X', 'Y')}

        dataset['X']['train'] = X[:idx]
        dataset['X']['test'] = X[idx:]
        dataset['Y']['train'] = Y[:idx]
        dataset['Y']['test'] = Y[idx:]

        if self.monotonic_x:

            sorted_indexes = np.argsort(dataset['X']['train'], axis=0)
            dataset['X']['train'] = dataset['X']['train'][sorted_indexes]
            dataset['Y']['train'] = dataset['Y']['train'][sorted_indexes]

            sorted_indexes = np.argsort(dataset['X']['test'])
            dataset['X']['test'] = dataset['X']['test'][sorted_indexes]
            dataset['Y']['test'] = dataset['Y']['test'][sorted_indexes]
        return dataset


class SklearnSplitter(BaseSplitter):

    def __init__(self):
        super().__init__()

    def split(self, X, Y):
        dataset = {el:{set_type:None for set_type in ('train', 'test')} for el in ('X', 'Y')}
        dataset["X"]['train'], dataset["X"]['test'], dataset["Y"]['train'], dataset["Y"]['test'] = train_test_split(X, Y, train_size=self.split_size, random_state=42)
        return dataset



class BaseDataset(ABC):

    dataset: dict[dict, np.ndarray]
    num_samples: int
    def __init__(self, monotonic_x = True):
        self.dataset = {el:{set_type:None for set_type in ('train', 'test')} for el in ('X', 'Y')}
        self.monotonic_x = monotonic_x
        self.splitter = SklearnSplitter()

    @abstractmethod
    def generate(self):
        ...

    def create_training_test_set(self, X, Y):
        return self.splitter.split(X, Y)
        


class LinearRegressionDataset(BaseDataset):

    slope: int
    intercept: int
    noise_mean: int
    noise_std: int

    domain_max: int
    def __init__(self, slope: int = 5, intercept: int = 10, noise_mean: int = 0, noise_std: int = 1, domain_max: int = 5, num_samples: int = 100):
        super().__init__()
        self.slope = slope
        self.intercept = intercept
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.num_samples = num_samples
        self.domain_max = domain_max


    def generate(self) -> dict[dict, np.ndarray]:

        if self.dataset['X']['train'] is None:
            noise = np.random.normal(loc=0, scale=self.noise_std, size=self.num_samples)
            X = np.linspace(0, self.domain_max, num=self.num_samples)            
            Y = self.slope * X + self.intercept + noise
            self.dataset = self.create_training_test_set(X, Y)
        return self.dataset

class PolynomialRegressionDataset(BaseDataset):

    filename: str
    def __init__(self, filename: str, x_name: str, y_name: str):
        super().__init__()
        self.filename = filename
        self.x = x_name
        self.y = y_name


    def generate(self) -> dict[dict, np.ndarray]:
        
        if self.dataset['X']['train'] is None:
            with open(self.filename) as data:
                df: pd.DataFrame = pd.read_csv(data)
                self.num_samples = len(df)
            self.dataset = self.create_training_test_set(df[self.x].to_numpy(), df[self.y].to_numpy())
        return self.dataset

class MultipleRegressionDataset(BaseDataset):

    def __init__(self):
        super().__init__(monotonic_x = False)


    def generate(self) -> dict[dict, np.ndarray]:
        
        from sklearn.datasets import fetch_california_housing

        if self.dataset['X']['train'] is None:
            data = fetch_california_housing()
            self.num_samples = len(data['data'])
            
            self.dataset = self.create_training_test_set(data['data'], data['target'])
            self.dataset['X']['feature_names'] = data["feature_names"] 
            self.dataset['Y']['feature_names'] = data["target_names"] 
        return self.dataset
