
import anuvada
import numpy as np
import torch
import pandas as pd

from anuvada.datasets.data_loader import CreateDataset
from anuvada.datasets.data_loader import LoadData
from anuvada.models.classification_cnn import ClassificationCNN


def run():
    data = CreateDataset()

    df = pd.read_csv('/home/sannu/Personal_Projects/data/labeledTrainData.tsv', encoding='utf-8', sep='\t')
    df.head()

    # passing only the first 512 samples, don't have a GPU here!
    y = list(df.sentiment.values)
    x = list(df.review.values)




if __name__ == '__main__':
    run()


