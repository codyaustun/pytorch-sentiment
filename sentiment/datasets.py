import os
import abc
from abc import ABCMeta

import pandas as pd
import torch.utils.data as data


class CachedDataset(data.Dataset):

    def __init__(self, dataset):
        """
        Cache transformed data from original dataset
        """
        self.dataset = dataset
        self._cache = [None] * len(dataset)

    def __getitem__(self, index):
        value = self._cache[index]
        if value is None:
            value = self.dataset[index]
            self._cache[index] = value

        return value

    def __len__(self):
        return len(self._cache)

    def __repr__(self):
        fmt_str = "***Cached***\n"
        fmt_str += self.dataset.__repr__()
        return fmt_str


class TextClassification(data.Dataset, metaclass=ABCMeta):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.train_data, self.train_labels = self.load_train_data()
        else:
            self.test_data, self.test_labels = self.load_test_data()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (doc, target) where target is index of the target class.
        """
        if self.train:
            doc, target = self.train_data[index], self.train_labels[index]
        else:
            doc, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            doc = self.transform(doc)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return doc, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))  # noqa: E501
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))  # noqa: E501
        return fmt_str

    @abc.abstractmethod
    def load_train_data(self):
        pass

    @abc.abstractmethod
    def load_test_data(self):
        pass

    @property
    @abc.abstractmethod
    def classes(self):
        pass


class AmazonReviewPolarity(TextClassification):
    def load_train_data(self):
        assert "amazon_review_polarity_csv" in self.root
        return self.load_data(os.path.join(self.root, 'train.csv'))

    def load_test_data(self):
        assert "amazon_review_polarity_csv" in self.root
        return self.load_data(os.path.join(self.root, 'test.csv'))

    def load_data(self, path):
        # TODO: Try csv module instead of pandas
        df = pd.read_csv(path, header=None,
                         names=['rating', 'subject', 'body'])
        df['subject'].fillna(value="", inplace=True)
        df['body'].fillna(value="", inplace=True)
        labels = (df['rating'] - df['rating'].min()).values
        data = (df['subject'] + " " + df['body']).values
        return data, labels

    @property
    def classes(self):
        return 2


class AmazonReviewFull(AmazonReviewPolarity):
    def load_train_data(self):
        assert "amazon_review_full_csv" in self.root
        return self.load_data(os.path.join(self.root, 'train.csv'))

    def load_test_data(self):
        assert "amazon_review_full_csv" in self.root
        return self.load_data(os.path.join(self.root, 'test.csv'))

    @property
    def classes(self):
        return 5
