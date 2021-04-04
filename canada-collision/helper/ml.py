# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model performance metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, auc, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve

class SupervisedModels:
  """
  Class for training and testing supervised ml models
  """

  def __init__(self):
    """
    Parameter initialization
    """

  def model_selection_cv(self, model, feat_mtx, labels, cv_fold, scoring=None):
    """
    Cross-validation on the training set

    Parameters
    ___________
    model: supervised classification model
    feat_mtx: feature matrix of the training set
    labels: class labels
    cv_fold: number of cross-validation fold
    scoring: performance metric

    Returns
    _____________
    Performance metrics on the cross-validation training set
    """

    # Fit the training set
    model.fit(feat_mtx, labels)

    # Compute accuracy on k-fold cross validation
    score = cross_val_score(model, feat_mtx, labels,cv=cv_fold, scoring=scoring)

    # Make prediction on k-fold cross validation
    y_val_pred = cross_val_predict(model, feat_mtx, labels, cv=cv_fold)

    # Make probability prediction on k-fold cross validation
    y_pred_proba = cross_val_predict(model, feat_mtx, labels,
     cv=cv_fold, method='predict_proba')[:, 1]

    # Print results
    print('-'* 75)
    print('Cross-validation accuracy (std): %f (%f)' % (score.mean(), score.std()))
    print('AUROC: %f' % (roc_auc_score(labels, y_pred_proba)))
    print('AUPRC: %f' % (average_precision_score(labels, y_pred_proba)))
    print('Predicted classes:', np.unique(y_val_pred))
    print('Confusion matrix:\n', confusion_matrix(labels, y_val_pred))
    print('Classification report:\n', classification_report(labels, y_val_pred))
    print('-'* 75)


  def test_prediction(self, model, feat_mtx, labels, test_feat_mtx, test_labels):
    """
    Predictions on the test set

    Parameters
    ___________
    model: supervised classification model
    feat_mtx: feature matrix of the training set
    test_feat_mtx: feature matrix of the test set
    labels: training set class labels
    test_labels: test set class labels

    Returns
    _____________
    Performance metrics on the test set
    """

    # Fit the training set
    model.fit(feat_mtx, labels)

    # Make prediction on the test set
    y_pred = model.predict(test_feat_mtx)

    # Compute the accuracy of the model
    accuracy = accuracy_score(test_labels, y_pred)

    # Predict probability
    y_pred_proba = model.predict_proba(test_feat_mtx)[:, 1]

    print('-'* 75)
    print('Test accuracy:  %f' % (accuracy))
    print('AUROC: %f' % (roc_auc_score(test_labels, y_pred_proba)))
    print('AUPRC: %f' % (average_precision_score(test_labels, y_pred_proba)))
    print('Predicted classes:', np.unique(y_pred))
    print('Confusion matrix:\n', confusion_matrix(test_labels, y_pred))
    print('Classification report:\n', classification_report(test_labels, y_pred))
    print('-'* 75)

  def plot_roc_prc_cv(self, model, feat_mtx, labels, color=None, label=None):
    """
    Plot ROC and PR curves for the cross-validation training set

    Parameters
    ___________
    model: supervised classification model
    feat_mtx: feature matrix of the training set
    labels: training set class labels
    color: matplotlib color
    label: matplotlib label

    Returns
    _____________
    Matplotlib line plot
    """

    # Fit the training set
    model.fit(feat_mtx, labels)

    # Make prediction on k-fold cross validation
    y_pred_proba = cross_val_predict(model, feat_mtx, labels, cv=5, method="predict_proba")

    # Compute the fpr and tpr for each classifier
    fpr, tpr, thresholds = roc_curve(labels, y_pred_proba[:, 1])

    # Compute the precisions and recalls for the classifier
    precisions, recalls, thresholds = precision_recall_curve(labels, y_pred_proba[:, 1])

    # Compute the area under the ROC curve for each classifier
    area_auc = roc_auc_score(labels, y_pred_proba[:, 1])

    # Compute the area under the PR curve for the classifier
    area_prc = auc(recalls, precisions)

    # ROC curve
    fig, (ax1,ax2) = plt.subplots(figsize = (20,8))
    ax1.plot(fpr, tpr, color=color, label=(label) % area_auc)
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.axis([0, 1, 0, 1])
    ax1.set_xlabel('False positive rate (FPR)')
    ax1.set_ylabel('True positive rate (TPR)')
    ax1.set_title('ROC Curve for the Cross-Validation Training Set')
    ax1.legend(loc='best')

    # PR curve
    ax2.plot(recalls, precisions, color=color, label=(label) % area_prc)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve for the Cross-Validation Training Set')
    ax2.legend(loc='best')

  def plot_roc_prc(self, model, feat_mtx, labels, test_feat_mtx, test_labels, color=None, label=None):
    """
    Plot ROC and PR curves for the test set

    Parameters
    ___________
    model: supervised classification model
    feat_mtx: feature matrix of the training set
    labels: training set class labels
    test_feat_mtx: feature matrix of the test set
    test_labels: test set class labels
    color: matplotlib color
    label: matplotlib label

    Returns
    _____________
    Matplotlib line plot
    """

    # Fit the model
    model.fit(feat_mtx, labels)

    # Predict probability
    y_pred_proba = model.predict_proba(test_feat_mtx)[:, 1]

    # Compute the fpr and tpr for each classifier
    fpr, tpr, thresholds = roc_curve(test_labels, y_pred_proba)

    # Compute the precisions and recalls for the classifier
    precisions, recalls, thresholds = precision_recall_curve(test_labels, y_pred_proba)

    # Compute the area under the ROC curve for each classifier
    area_auc = roc_auc_score(test_labels, y_pred_proba)

    # Compute the area under the PR curve for the classifier
    area_prc = auc(recalls, precisions)

    # ROC curve
    fig, (ax1,ax2) = plt.subplots(figsize = (20,8))
    ax1.plot(fpr, tpr, color=color, label=(label) % area_auc)
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.axis([0, 1, 0, 1])
    ax1.set_xlabel('False positive rate (FPR)')
    ax1.set_ylabel('True positive rate (TPR)')
    ax1.set_title('ROC Curve for the Test Set')
    ax1.legend(loc='best')

    # PR curve
    ax2.plot(recalls, precisions, color=color, label=(label) % area_prc)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve for the Test Set')
    ax2.legend(loc='best')

