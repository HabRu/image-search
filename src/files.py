import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors
from keras.applications.vgg19 import preprocess_input
import scipy.sparse as sp

def save(arr, filename):
    with open(filename, 'w') as fd:
        fd.write(','.join(arr))

def load_sparse_matrix(filename):
    y = np.load(filename)
    z = sp.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'])
    return z

def save_sparse_matrix(filename, x):
    x_coo = x.tocoo()
    row = x_coo.row
    col = x_coo.col
    data = x_coo.data
    shape = x_coo.shape
    np.savez(filename, row=row, col=col, data=data, shape=shape)

def upload_image(img):
    X = np.zeros((1, 224, 224, 3))
    
    img = img.resize((224,224))
    img_array = np.array(img)

    print(img_array.shape)

    X[0, :, :, :] = img_array
    
    base_model = tf.keras.models.load_model('./data/my_model.h5', compile=False)
    # Read about fc1 here http://cs231n.github.io/convolutional-networks/
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    result = model.predict(preprocess_input(X))

    vecs = load_sparse_matrix("./data/result.npz")

    filenames = open('./data/files', 'r').readline().split(',')

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(vecs)

    dist, indices = knn.kneighbors(result.ravel().reshape(1, -1), n_neighbors=6)
    dist, indices = dist.flatten(), indices.flatten()
    
    files = [((file[1:], file[1:]), d) for file, d in [(filenames[indices[i]], dist[i]) for i in range(len(indices))]]

    return files