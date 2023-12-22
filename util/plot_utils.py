import itertools
import matplotlib.pyplot as plt
import numpy as np

from time import time
from sklearn.manifold import TSNE


embeddings_dict = {
    't-SNE embedding': TSNE(
        n_components=2,
        n_iter=500,
        n_iter_without_progress=150,
        n_jobs=2,
        random_state=0,
    )
}

def get_plot_features(X, y=None, embedding_name='t-SNE embedding', return_time=False):
    if embedding_name.startswith("Linear Discriminant Analysis"):
        data = X.copy()
        data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
    else:
        data = X
    print(f"Computing {embedding_name}...")
    start_time = time()
    if y is None:
        projection = embeddings_dict[embedding_name].fit_transform(data)
    else:
        projection = embeddings_dict[embedding_name].fit_transform(data, y)
    timing = time() - start_time
    if return_time:
        return projection, timing
    else:
        return projection
    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.rcParams['figure.figsize'] = [10,7]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()