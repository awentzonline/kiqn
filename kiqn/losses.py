import keras.backend as K


def iqn_loss(num_quantiles, k=1.):
    """
    Expects the `pred` tensor to be a batch-wise concatenation
    of

    I'm sure I adapted the Huber loss part of this from
    somewhere a while back when I started experimenting
    with quantile networks, but don't remember where.
    """
    def f(target, pred):
        batch_size = K.shape(target)[0]
        output_dims = K.shape(target)[-1]
        #num_quantiles = K.shape(pred)[0] // batch_size
        # reshape
        qs = pred[:batch_size * num_quantiles]
        pred = pred[batch_size * num_quantiles:]
        qs = K.reshape(qs, (-1, batch_size, output_dims))
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
        # Take sum of loss over quantile dim and then batch mean
        return K.mean(K.sum(K.abs(qs - K.cast(err < 0, K.floatx())) * loss, axis=0))
    return f
