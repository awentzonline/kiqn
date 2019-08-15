from keras import backend as K, layers, models
import numpy as np

from kiqn.layers import IQN
from kiqn.losses import iqn_loss


np.random.seed(0xbeef)

input_dims = 3
output_dims = 2
num_quantiles = 8
quantile_embedding_dims = 16
feature_dims = 32
batch_size = 32
epochs = 100

# build model
# input feature extractor
x_input = layers.Input(batch_shape=(batch_size, input_dims,), name='x_in')
x = layers.Dense(feature_dims, activation='relu', name='x_feats')(x_input)

# Condition the input features on quantile embeddings
x_tau, taus = IQN(num_quantiles=num_quantiles,
                  embedding_dims=quantile_embedding_dims,
                  name='iqn')(x)

# Tile the quantile samples so they can be concatenated with the outputs
taus = layers.Lambda(lambda x: K.tile(taus, (1, output_dims)), name='taus')(taus)
p_output = layers.Dense(output_dims, name='p_out')(x_tau)
iqn_output = layers.concatenate([taus, p_output], axis=0, name='iqn_out')

model = models.Model(inputs=[x_input],
                     outputs=[iqn_output])
loss = iqn_loss(num_quantiles=num_quantiles)

model.compile(optimizer='adam', loss=loss)
model.summary()

# dumb dataset
dataset_size = 100000
means = np.arange(input_dims)
stds = np.random.rand(input_dims)
x = np.random.randn(dataset_size, input_dims)
x = x * stds + means

weights = np.random.uniform(-1, 1, (input_dims, output_dims))
bias = np.random.uniform(-1, 1, output_dims)
y = np.dot(x, weights) + bias
print(stds, means, weights, bias)

try:
    model.fit([x], [y], batch_size=batch_size, epochs=epochs)
except KeyboardInterrupt:
    pass

f_output = K.function(
    [model.inputs[0]],
    [model.get_layer('p_out').output]
)
p_out = f_output([x[:batch_size], 1.])[0]
p_out = np.reshape(p_out, (num_quantiles, batch_size, output_dims)).mean(axis=0)
target = y[:batch_size]
err = np.abs(target - p_out)
err_ratio = np.abs(err / target)
print('Err', err_ratio, err)
