import keras.backend as K


def masked_global_average_pooling_1d(args):
    x, mask = args
    # x: none, steps, features
    # mask: none, steps
    feature_dim = K.int_shape(x)[-1]
    mask_tile = K.expand_dims(mask, axis=-1)
    mask_tile = K.tile(mask_tile, (1, 1, feature_dim))
    x = x * mask_tile
    return K.mean(x, axis=1)


def masked_global_max_pooling_1d(args):
    x, mask = args
    # x: none, steps, features
    # mask: none, steps
    feature_dim = K.int_shape(x)[-1]
    mask_tile = K.expand_dims(mask, axis=-1)
    mask_tile = K.tile(mask_tile, (1, 1, feature_dim))
    x = x * mask_tile
    return K.max(x, axis=1)