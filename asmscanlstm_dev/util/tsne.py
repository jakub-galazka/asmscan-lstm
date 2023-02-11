import numpy as np
from sklearn.manifold import TSNE


def tsne_2d(mdim_representation: any) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    '''
        T-distributed Stochastic Neighbor Embedding
        https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        
        Parameters
        ----------
        mdim_representation : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Multidimensional representation.

        Returns
        -------
        Embedding of the multidimensional representation in 2D space.

        x : np.ndarray[np.float64]
            X dimension.

        y : np.ndarray[np.float64]
            Y dimension.
    '''
    tsne = TSNE(verbose=1).fit_transform(mdim_representation)
    return tsne[:, 0], tsne[:, 1] # x, y
