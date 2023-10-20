import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


class FeatureVisualize(object):
    '''
    Visualize features by TSNE
    '''

    def __init__(self, features, labels):
        '''
        features: (m,n)
        labels: (m,)
        '''
        self.features = features
        self.labels = labels

    def plot_tsne(self, save_eps=False):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        tsne = TSNE(n_components=2, init='pca', random_state=2022)
        features = tsne.fit_transform(self.features)
        data = features

        for i in range(data.shape[0]):
            plt.scatter(data[i, 0], data[i, 1],s=10,color=plt.cm.Paired(self.labels[i]))
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        
        # plt.xticks([])
        # plt.yticks([])
        plt.title('T-SNE')
        if save_eps:
            plt.savefig('tsne.eps', dpi=600, format='eps')
        plt.show()
