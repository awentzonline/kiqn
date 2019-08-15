kiqn
====
[Implicit Quantile Network](https://arxiv.org/abs/1806.06923) layer for Keras. Could use some work--PRs welcome.

Usage
=====
```
from kiqn.layers import IQN
from kiqn.losses import iqn_loss

...your good codes...

# `x_tau` contains the input features conditioned
# on the quantile embeddings
# `taus` are the quantile samples
x_tau, taus = IQN(num_quantiles=32,
                  embedding_dims=64,
                  name='iqn')(input_features)

...more good codes...

# reshape `taus` so they can be concatenated with the output
taus = Lambda(lambda x: K.tile(taus, (1, output_dims)), name='taus_rehaped')(taus)

# concatenate along batch axis, taus first
iqn_output = layers.concatenate([taus, p_output], axis=0, name='iqn_out')

...10x codes...

model.compile(optimizer='adam', iqn_loss(num_quantiles=num_quantiles))
```
