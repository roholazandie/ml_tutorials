from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import manifold, datasets
import numpy as np

import plotlyvisualization as plotly


def incremental_pca(X):
    n_batches = 100
    inc_pca = IncrementalPCA(n_components=54)
    for X_batch in np.array_split(X, n_batches):
        inc_pca.partial_fit(X_batch)

    X_reduced = inc_pca.transform(X)
    X_recovered = inc_pca.inverse_transform(X_reduced)
    plt.imshow(X_recovered[0].reshape(28, 28))
    plt.show()


def pca_for_compression(X):
    '''
    in this function we try to reduce the dimensionality
    and then recover to the original image
    :return:
    '''
    pca = PCA(n_components=154)
    X_reduced = pca.fit_transform(X)
    X_recovered = pca.inverse_transform(X_reduced)
    plt.imshow(X_recovered[0].reshape(28, 28))
    plt.show()





def kernel_pca(X):
    k_pca = KernelPCA(n_components=10, kernel='rbf', gamma=0.01, fit_inverse_transform=True)
    X_reduced = k_pca.fit_transform(X)
    X_recovered = k_pca.inverse_transform(X_reduced)
    plt.imshow(X_recovered[0].reshape(28, 28))
    plt.show()
    plt.imshow(X[0].reshape(28, 28))
    plt.show()



def draw_kernel_pca(X):
    k_pca = KernelPCA(n_components=10, kernel='sigmoid', gamma=0.001)
    #pca = PCA(n_components=100)
    X_reduced = k_pca.fit_transform(X)
    #plotly.scatter(X_reduced[:,0], X_reduced[:,1])
    plotly.scatter3(X_reduced[:,0], X_reduced[:,1], X_reduced[:,2], color=X_reduced[:,2])



def tunning_hyperparameters():
    mnist = input_data.read_data_sets("../MNIST-data")
    X_test = mnist.train.images
    y_test = mnist.train.labels

    clf = Pipeline([("kpca", KernelPCA(n_components=2)),
                    ("log_reg", LogisticRegression())])

    param_grid = [{"kpca__gamma": np.linspace(0.03, 0.05, 10),
                   "kpca__kernel": ["rbf", "sigmoid"]}]

    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(X_test, y_test)
    print(grid_search.best_params_)
    #{'kpca__gamma': 0.029999999999999999, 'kpca__kernel': 'sigmoid'}


def unsupervised_tunning_hyperparameters():
    mnist = input_data.read_data_sets("../MNIST-data")
    X_test = mnist.test.images

    best_reconstruction_error = 100000
    best_params = {'gamma':0, 'kernel':'rbf'}
    for kernel in ('sigmoid', 'rbf'):
        for gamma in np.linspace(0.03, 0.05, 10):
            print(gamma)
            k_pca = KernelPCA(n_components=2, kernel=kernel, gamma=gamma, fit_inverse_transform=True)
            X_reduced = k_pca.fit_transform(X_test)
            X_reconstructed = k_pca.inverse_transform(X_reduced)
            reconstruction_error = mean_squared_error(X_reconstructed, X)
            if reconstruction_error < best_reconstruction_error:
                best_reconstruction_error = reconstruction_error
                best_params['gamma'] = gamma
                best_params['kernel'] = kernel

    print(best_reconstruction_error)
    print(best_params) #{'gamma': 0.029999999999999999, 'kernel': 'rbf'}



def plot_swiss_roll():
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=1500)
    plotly.scatter3(X[:,0], X[:,1], X[:,2], color)


def lle_swiss_roll():
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=1500)
    lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
    X_reduced = lle.fit_transform(X)

    plotly.scatter(X_reduced[:,0], X_reduced[:,1], color)




if __name__ == "__main__":
    # Data
    mnist = input_data.read_data_sets("../MNIST-data")
    X_test = mnist.test.images
    y_test = mnist.test.labels

    X = mnist.test.images

    #pca_for_compression(X)

    #incremental_pca(X)

    #draw_kernel_pca(X)

    #unsupervised_tunning_hyperparameters()

    #plot_swiss_roll()

    lle_swiss_roll()