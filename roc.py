import pandas as pd
import numpy as np
import math
from scipy.stats import rankdata
from random import randint
import matplotlib.pyplot as plt 
#from collections import OrderedDict

def roc_coordinates(classes, scores, roc_classes):
    #categories = np.array(list( {k:1 for k in classes} ))
    #for so that we start with the heigh score (mostly healthy )
    ix_sort    = np.argsort(-scores)
    classes    = classes[ix_sort]
    scores     = scores[ix_sort]
    #take the first class as good (healthy = heigh scores) and the rest as bad(sick=lower scores)
    cat = roc_classes[0]

    n_samples = len(scores)
    n_pos     = np.sum(classes == cat)
    n_neg     = n_samples - n_pos

    tp = np.cumsum(classes==cat)/n_pos
    fp = np.cumsum((classes==cat)==False)/n_neg
    tn = np.cumsum(classes[::-1]==cat)/n_pos
    fn = np.cumsum((classes[::-1]==cat)==False)/n_neg
    recall    = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)    
    df = pd.DataFrame( columns = 
                       ["category", 
                        "score",
                        "true_positive",
                        "false_positive",
                        "true_negative",
                        "false_negative",
                        "recall",
                        "precision",
                        "f1",
                       ])
    df.category       = classes
    df.score          = scores
    df.true_positive  = np.round(tp,3)
    df.false_positive = np.round(fp,3)
    df.true_negative  = np.round(tn,3)
    df.false_negative = np.round(fn,3)
    df.recall         = np.round(recall,3)
    df.precision      = np.round(precision,3)
    df.f1             = np.round(f1,3)

    """
    #the following does not work due to a bug in pandas
    df = pd.DataFrame( OrderedDict( \
                       [("category", classes),\
                        ("score", scores),
                        ("true_positive",np.round(tp,2)),\
                        ("false_positive",np.round(fp,2)),\
                        ("true_negative",np.round(tn,2)),\
                        ("false_negative",np.round(fn,2)),\
                        ("recall",np.round(recall,2)),\
                        ("precision",np.round(precision,2)),\
                        ("f1",np.round(f1,2))\
                       ]))
    """

    return df, roc_classes



def auc_by_roc_coordinates(fp, tp):
    """
    (adapted from sklearn)
    Compute Area Under the Curve (AUC) using the trapezoidal rule
    This is a general function, given points on a curve.
    Parameters
    ----------
    x : array, roc_fpr False Positive Rate, shape = [n], x coordinates.
    y : array, roc_tpr True Positive Rate, shape = [n], y coordinates.
    Returns
    -------
    auc : float
    """
    assert len(fp) == len(tp)
    
    fp = np.array(fp)
    tp = np.array(tp)

    #must start at zero to get th right area
    if False == (fp[0] == 0 and tp[0] == 0):
        np.insert(fp,0,0)
        np.insert(tp,0,0)

    if fp.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % fp.shape)

    direction = 1
    dx = np.diff(fp)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("The x array is not increasing: %s" % fp)

    area = direction * np.trapz(tp, fp)
    if area < 0.50:
        area = 1.0 - area

    return area

#calculates the AUC directly using the mannwhitney approach
def auc_by_utest(classes, scores):
    categories = np.array(list( {k:1 for k in classes} ))
    rank       = rankdata(scores)
             
    #just pick the first class for now.We could go over alle classes and calculate an auc for each
    #for c in classes:
    c = categories[0]
    #this should be in the loop and auc should be a list
    labels = classes==c

    n1  = np.sum(labels)
    n2  = np.sum(labels==False)
    R1  = np.sum(rank[labels])
    U1  = R1 - n1 * (n1 + 1)/2
    auc = U1/(n1 * n2)
    if auc < 0.50:
        auc = 1.0 - auc

    return auc

def plot_pdf(bins, class_predictions, colors, ax):
    ax.set_xlim([min(bins),max(bins)])
    
    i=0
    for c,v in class_predictions.items():
        ax.hist( v, bins, color=colors[i], alpha=0.5)
        i+=1
        
    #ax.set_ylim([0,max(histograms.flatten)])
    ax.set_title("Probability Distribution", fontsize=12)
    ax.set_ylabel('Counts', fontsize=10)
    ax.set_xlabel('Value slices', fontsize=10)
    ax.legend(list(class_predictions.keys()))

def plot_roc(score, tpr, fpr, cat, ax, thresh_decimals=0):
    nb_thresholds=10
    threshold = np.linspace(np.min(score), np.max(score), nb_thresholds)
    tp_thresh = np.interp(-threshold, -score, tpr)
    fp_thresh = np.interp(-threshold, -score, fpr)
    threshold = np.round(threshold,thresh_decimals).astype(int if thresh_decimals==0 else float)
    #print(threshold)
    #print(np.round(tp_thresh3),)
    #print(np.round(fp_thresh,3)
    
    #Plotting final ROC curve
    ax.plot(fpr, tpr, color="b")
    x = np.linspace(0,1,3)    
    ax.plot(x,x, "ro--")
    
    #plot thresholds labels near but not on the roc
    for i in range(len(threshold)):
        ax.text(fp_thresh[i] - 0.07,tp_thresh[i] + 0.01, threshold[i], color="b", fontdict={'size': 8});
    ax.scatter(fp_thresh, tp_thresh, s=10, color="b")    
    
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_title("ROC Curve", fontsize=14)
    ax.set_xlabel('FPR', fontsize=12)
    ax.tick_params(direction='out', pad=20)
    ax.set_ylabel('TPR', fontsize=12)
    #ax.grid()
    auc = auc_by_roc_coordinates(tpr,fpr)
    ax.legend([f"AUC for class \"{cat}\": {auc:.2f}"]) 