kiqn
====
[Implicit Quantile Network](https://arxiv.org/abs/1806.06923) layer for Keras. Could use some work--PRs welcome.

Usage
=====
```
from kiqn.layers import IQN
from kiqn.losses import iqn_loss

...your good codes...

x_in = Input(batch_shape=(batch_size, input_dims,))
y_in = Input(batch_shape=(batch_size, output_dims,))

...more good codes...

# `x_tau` contains the input features conditioned on the quantile embeddings
# `taus` are the quantile samples
x_tau, taus = IQN(
    num_quantiles=32,
    embedding_dims=64,
    name='iqn'
)(x_features)

... y_out = f(x_tau) ...

model = Model(inputs=[x_in, y_in], outputs=[y_out])
model.add_loss(iqn_loss(y_in, y_out, taus))
model.compile(optimizer='adam')

# Since we don't assign loss functions in `compile`, don't send any targets
model.fit([train_x, train_y], [])

# Use it
f_output = K.function(
    [model.inputs[0]],
    [model.get_layer('y_out').output]
)
```
