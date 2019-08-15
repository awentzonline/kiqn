import keras.backend as K
from keras.layers import Dense, Layer
import numpy as np


class IQN(Layer):
    def __init__(self, num_quantiles=8, embedding_dims=32, *args, **kwargs):
        self.num_quantiles = num_quantiles
        self.embedding_dims = embedding_dims
        super().__init__(*args, **kwargs)

    def build(self, features_shape):
        batch_size = features_shape[0]
        feature_shape = features_shape[1:]
        flat_flature_dims = np.prod(feature_shape)
        self.tau_rng = K.random_uniform((batch_size * self.num_quantiles, 1))
        # TOOD: DCT layer
        self.tau_embedding = Dense(flat_flature_dims, activation='tanh')
        super().build(features_shape)

    def call(self, features):
        feature_shape = K.shape(features)[1:]
        tau_samples = self.tau_rng
        # shape (batch_size * num_quantiles, embedding_dims)
        tau_samples_tiled = K.tile(tau_samples, [1, self.embedding_dims])
        tau_embedding = self.tau_embedding(tau_samples_tiled)
        tiled_features = K.tile(features, [self.num_quantiles, 1])
        return [tau_embedding * tiled_features, tau_samples]

    def compute_output_shape(self, features_shape):
        batch_size, feature_shape = features_shape[0], features_shape[1:]
        batch_qs = batch_size * self.num_quantiles
        return [(batch_qs,) + feature_shape, (batch_qs, 1)]

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            num_quantiles=num_quantiles,
            embedding_dims=embedding_dims,
        ))
        return config
