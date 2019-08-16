import keras.backend as K


def iqn_loss(target, pred, qs, k=1.):
    """
    I'm sure I adapted the Huber loss part of this from
    somewhere a while back when I started experimenting
    with quantile networks, but don't remember where.
    """
    batch_size = K.shape(target)[0]
    output_dims = K.shape(target)[-1]
    num_quantiles = K.shape(pred)[0] // batch_size
    # reshape
    qs = K.reshape(qs, (-1, batch_size, 1))
    pred = K.reshape(pred, (-1, batch_size, output_dims))
    target = K.tile(target, [num_quantiles, 1])
    target = K.reshape(target, (-1, batch_size, output_dims))

    # huber quantile loss
    err = target - pred
    loss = K.switch(
        K.abs(err) <= k,
        0.5 * K.square(err),
        k * (K.abs(err) - 0.5 * k)
    )
    print('errr', list(map(lambda x: K.int_shape(x), (loss, err, qs, target, pred, K.cast(err < 0, K.floatx())))))
    # Take sum of loss over quantile dim and then batch mean
    return K.mean(K.sum(K.abs(qs - K.cast(err < 0, K.floatx())) * loss, axis=0))
