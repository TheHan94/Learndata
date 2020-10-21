import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from pathlib import Path
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.model_selection import GridSearchCV, PredefinedSplit

########################################
##### Definition of the functions #####
########################################
def SMAPE(y_true, y_pred, perc=True):
    """
    This function computes the symmetric mean absolute percentage error given two arrays
    """
    assert len(y_true) == len(y_pred)
    smape_ = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    if perc:
        return 100 * smape_
    else:
        return smape_
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.colors,
                          ax=None, label='label'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    def format_func(value, tick_number):
        cl_map = dict(zip(range(len(classes)), classes))
        return cl_map.get(value, 'XX')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm.T, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.xaxis.set_tick_params(rotation=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(i, j, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('Predicted {}'.format(label))
    ax.set_xlabel('True {}'.format(label))

#######################################
##### Definition of the parameters #####

SENSORS = {
    'all': ['G1-1', 'G1-2', 'G1-3', 'G1-4', 'G2-1', 'G2-2', 'G2-3', 'G2-4'],
    's1': ['G1-1', 'G1-2', 'G1-3', 'G1-4'],
    's2': ['G2-1', 'G2-2', 'G2-3', 'G2-4']
}
SPLITS_TO_CONSIDER = [30, 31, 32, 33, 34, 35, 36, 40, 41, 42, 43]
EVAL_SCORE = SMAPE
USED_SENSORS = 'all'
POWER = 2.
DPI = 600
# Parameters grid for classifiers
parameters_clf = {
    'C': [0.01, 0.1, 1, 2, 5, 10, 50, 100],
    'fit_intercept': [True, False]
}
# Parameters grid for regressors
parameters_reg = {
    'C': [0.01, 0.1, 1, 2, 5, 10, 50, 100],
    'epsilon': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'fit_intercept': [False]
}
np.random.seed(1)
palette = np.array(sns.color_palette())

########################################
########### Data preparation ###########

data = pd.read_csv('gas_detection_data.csv', skiprows=1, index_col=0)

classes = sorted(set(data['Gas']))
idx_by_class = {c: (data.index[data['Gas'] == c] - 1).tolist() for c in classes}
y = np.array([classes.index(c) for c in data['Gas'].values])
conc = data['Conc'].values

########################################
############ Model training ############

for s in SPLITS_TO_CONSIDER:
    print('Split{} Sensor {}'.format(s if s > 0 else '', USED_SENSORS))
    split_col = data['Split{}'.format(s if s >= 0 else '')]

    train_idx = np.where(split_col == 'Tr')[0]
    val_idx = np.where(split_col == 'V')[0]
    in_idx = np.where(split_col == 'In')[0]
    ex_idx = np.where(split_col == 'Ex')[0]

    if len(val_idx) == 0:
        val_idx = train_idx

    X = data[SENSORS[USED_SENSORS]].values - 1

    train_or_val_idx = sorted(set(train_idx).union(set(val_idx)))
    test_idx = sorted(set(in_idx).union(set(ex_idx)))

    X_tv = X[train_or_val_idx]
    y_tv = y[train_or_val_idx]

    conc_tv = conc[sorted(train_or_val_idx)]

    # Build the predefined splits for the cross validation
    ps_clf = PredefinedSplit(test_fold=[0 if i in val_idx else -1 for i in sorted(train_or_val_idx)])
    ps_reg = [PredefinedSplit(
        test_fold=[0 if k in val_idx else -1 for k in sorted(set(idx_by_class[c]).intersection(set(train_or_val_idx)))]
    ) for c in classes]

    # Construct grid search cross validation to select the best classifier given the validation set
    clf = GridSearchCV(
        LinearSVC(max_iter=1e9),
        parameters_clf,
        scoring='accuracy',
        refit=True,
        cv=list(ps_clf.split())
    )

    # Construct grid search cross validation to select the best regressors given the validation set
    reg = [
        GridSearchCV(
            LinearSVR(loss='squared_epsilon_insensitive', max_iter=1e9),
            parameters_reg,
            cv=list(ps_reg[i].split()),
            scoring=make_scorer(EVAL_SCORE, greater_is_better=False),
            n_jobs=4,
            refit=True)
        for i, _ in enumerate(classes)
    ]

    # Train the classifier model
    clf.fit(X_tv, y_tv)
    #clf.score(X_tv, y_tv)
    #print('Validation scores classifiers', clf.score(X_tv, y_tv))

    #y_pred = clf.predict(X[test_idx])
    #y_pred_train = clf.predict(X[train_idx])
    #y_pred_val = clf.predict(X[val_idx])