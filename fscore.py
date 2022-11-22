import numpy as np
import scipy.io as scio
import sklearn.cluster as skc
import sklearn.metrics as skm

# load the array

imIF = np.array(scio.loadmat("./062B6.mat")["imIF"])
imIF = imIF.reshape((imIF.shape[0] * imIF.shape[1], imIF.shape[2]))

# kmeans = skc.KMeans(n_clusters=4, init="k-means++").fit(imIF)

# load ground truths

imL = np.array(scio.loadmat("./062B6_gt.mat")["imL"])
imL = imL.reshape(imL.shape[0] * imL.shape[1])


# group the ground truths

def grouping(a: list) -> list:
    out = []
    for point in a:
        out.append([a.index(p) for p in a if p==point])
        a = [i for i in a if i!=point]
    return out

def calculate(pred: list, true: list) -> list:
    out = []
    for pred_label in pred:
        out.append([skm.f1_score(true_label, pred_label, average="macro") for true_label in true])
