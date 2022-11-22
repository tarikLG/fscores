import numpy as np
import scipy.io as scio
import sklearn.cluster as skc
import sklearn.metrics as skm

class KmeansAccuracy:
    def __init__(self) -> None:
        # load the data

        self.imIF = np.array(scio.loadmat("./062B6.mat")["imIF"])
        self.imIF = self.imIF.reshape((self.imIF.shape[0] * self.imIF.shape[1], self.imIF.shape[2]))

        # load ground truths

        self.imL = np.array(scio.loadmat("./062B6_gt.mat")["imL"])
        self.imL = self.imL.reshape(self.imL.shape[0] * self.imL.shape[1])

        # do kmeans clustering
 
        self.kmeans = skc.KMeans(n_clusters=2, init="k-means++").fit(self.imIF)

    # group labels with each other based and rename classes into indices and labels into colums

    def grouping(a: list) -> list:
        out = []
        for point in a:
            out.append([a.index(p) for p in a if p==point])
            a = [i for i in a if i!=point]
        return out

    # calculate f1scores

    def calculate(pred: list, true: list) -> list:
        out = []
        for pred_label in pred:
            out.append([skm.f1_score(true_label, pred_label, average="macro") for true_label in true])


ka = KmeansAccuracy()

grouped_gt = ka.grouping(ka.imL)
grouped_predicted = ka.grouping(ka.kmeans)

f1_scores = ka.calculate(grouped_predicted, grouped_gt)

print(f1_scores)
