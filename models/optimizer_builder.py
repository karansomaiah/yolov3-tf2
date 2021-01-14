import tensorflow as tf
from protos import train_pb2


def build(config):
    optimizer_config_map = {
        'adadelta_optimizer': tf.keras.optimizers.Adadelta,
        'adagrad_optimizer': tf.keras.optimizers.Adagrad,
        'adam_optimizer': tf.keras.optimizers.Adam,
        'nadam_optimizer': tf.keras.optimizers.Nadam,
        'rmsprop_optimizer': tf.keras.optimizers.RMSprop,
        'sgd_optimizer': tf.keras.optimizers.SGD
    }

    learning_rate_scheduler_map = {
        'exponential_decay': tf.keras.optimizers.schedules.ExponentialDecay,
        'inversetime_decay': tf.keras.optimizers.schedules.InverseTimeDecay,
        'piecewise_constant_decay':
        tf.keras.optimizers.schedules.PiecewiseConstantDecay,
        'polynomial_decay': tf.keras.optimizers.schedules.PolynomialDecay,
        'constant': float,
    }

    optimizer_string = config.WhichOneof('optimizer')
    learning_rate_string = config.WhichOneof('learning_rate')

    # create the optimizer_pb
    optimizer_pb = getattr(config, optimizer_string)
    optimizer_args = {
        field.name: getattr(optimizer_pb, field.name)
        for field in optimizer_pb
    }

    # create lr pb
    if learning_rate_string != "constant":
        learning_rate_pb = getattr(config, learning_rate_string)
        learning_rate_args = {
            field.name: getattr(learning_rate_pb, field.name) for field
    }
    else:
        learning_rate_pb = getattr(config, learning_rate_string)
        learning_rate_args = learning_rate_pb.learning_rate
    
    # create Learning Rate Scheduler
    learning_rate_scheduler = \
            learning_rate_scheduler_map[learning_rate_string](learning_rate_args)

    # set the optimizer learning rate argument
    optimizer_args['learning_rate'] = learning_rate_scheduler
    
    # create the optimizer
    optimizer = optimizer_config_map[optimizer_string](learning_rate_args)

    return optimizer

