import os
import numpy as np
import tensorflow as tf
from scipy import interpolate

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X):

        # triple the number of points in the sequence to increase the resolution by a factor of 3
        x_lin = np.linspace(0, 35, 36 * 3)

        # new augmented dataset
        X_spline = np.zeros((X.shape[0], x_lin.size, 6))
        
        for i in np.arange(X.shape[0]): # for each sample
            for j in np.arange(6): # for each feature
                # add a cubic spline interpolation between the data points of X[sample, :, 0]
                interpolation = interpolate.interp1d(np.arange(0, 36), X[i, :, j], kind='cubic', fill_value="extrapolate")
                X_spline[i, :, j] = interpolation(x_lin)
        
        out = self.model.predict(X_spline)
        out = tf.argmax(out, axis=-1)

        return out