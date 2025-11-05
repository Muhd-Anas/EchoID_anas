from src.utils.config_utils import read_config
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.callbacks import EarlyStopping, ReduceLROnPlateau


conf = read_config()


def get_callbacks():
    """Create a list of Keras callbacks based on the config.yaml configuration.

    Returns:
        List[keras.callbacks.Callback]: A list of Keras callbacks.
    """
    callbacks = [
        
        # -------------------Callbacks-------------------
        EarlyStopping(
            monitor=conf['training']['early_stopping']['monitor'],
            patience=conf['training']['early_stopping']['patience'],
            verbose=1,
            restore_best_weights=conf['training']['early_stopping'].get('restore_best_weights', True),
            start_from_epoch=conf['training']['early_stopping'].get('start_from_epoch', 0),
        ),

        # -------------------Learning Rate Scheduler-------------------
        ReduceLROnPlateau(
            monitor=conf['training']['reduce_lr']['monitor'],
            factor=conf['training']['reduce_lr']['factor'],
            patience=conf['training']['reduce_lr']['patience'],
            min_lr=conf['training']['reduce_lr']['min_lr'],
            verbose=1
        )
    ]

    return callbacks
