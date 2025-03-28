import pandas as pd
import os
import numpy as np
import torch

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit, GroupShuffleSplit
from tsai.all import *
from fastai.metrics import RocAucBinary

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import fastai.learner as fastail
from fastai.callback.tracker import ReduceLROnPlateau, EarlyStoppingCallback
from fastai.callback.training import GradientAccumulation, ShortEpochCallback
from fastai.distributed import ParallelTrainer, DistributedTrainer, DistributedDL
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from fastai.metrics import BalancedAccuracy, RocAucBinary


class TSC(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=42, epochs=100):
        self.model = None
        self.classes_ = np.array([0, 1])
        self.random_state = random_state
        self.epochs = epochs
        self.batch_tfms = [TSStandardize(by_sample=False)]
        self.tfms = [None, TSClassification()]

    def fit(self, X, y, groups):
        for sp0, sp1 in StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=self.random_state).split(X, y, groups=groups): # 8 splits
            break

        patients = 5
        self.model = TSClassifier(X, y,
                                  arch="InceptionTime", tfms=self.tfms, # path=f"./models/{self.random_state}",
                                  metrics=[BalancedAccuracy(), RocAucBinary()],
                                  train_metrics=True,
                                  splits=(sp0, sp1),
                                  bs=64,
                                  cbs=[
                                      EarlyStoppingCallback(min_delta=0.0, patience=patients, comp=np.less),
                                      SaveModel(),
                                      ReduceLROnPlateau(monitor='valid_loss', comp=np.less, patience=2, factor=10, min_lr=0 ,reset_on_fit=True)
                                  ],
                                  ### -> if you want a propper balanced accuracy
                                  # loss_func=CrossEntropyLossFlat(reduction="mean", weight=torch.Tensor((dist[1]/dist.sum(), dist[0]/dist.sum())).to("cuda")),
                                  batch_tfms=self.batch_tfms,
                                  lr=5e-4,
                                  verbose=True)

        self.model.fit(self.epochs)

        return self

    def fine_tune(self, X, y, groups, path):
        for sp0, sp1 in StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=self.random_state).split(X, y, groups=groups):
            break
        patients = 5
        self.model = TSClassifier(X, y,
                                  arch="InceptionTime", tfms=self.tfms, # path=f"./models/{self.random_state}",
                                  metrics=[BalancedAccuracy(), RocAucBinary()],
                                  train_metrics=True,
                                  splits=(sp0, sp1),
                                  pretrained=True,
                                  weights_path=path,
                                  bs=64,
                                  cbs=[
                                      EarlyStoppingCallback(min_delta=0.0, patience=patients, comp=np.less),
                                      SaveModel(),
                                  ],
                                  ### -> if you want a propper balanced accuracy
                                  # loss_func=CrossEntropyLossFlat(reduction="mean", weight=torch.Tensor((dist[1]/dist.sum(), dist[0]/dist.sum())).to("cuda")),
                                  # loss_func=FocalLossFlat(),
                                  batch_tfms=self.batch_tfms,
                                  exclude_head=False,
                                  lr=1e-4,
                                  verbose=True)
        self.model.fit(self.epochs)

    def predict(self, X, bs=64):
        preds_int = self.model.get_X_preds(X, bs=bs)[2].astype(int).astype(int)
        return preds_int

    def predict_proba(self, X, bs=64):
        preds = self.model.get_X_preds(X, bs=bs)[0].numpy()[:, :]
        return preds

    def export(self, path):
        self.model.export(path)

    def load(self, path):
        self.model = load_learner(path, cpu=False)
