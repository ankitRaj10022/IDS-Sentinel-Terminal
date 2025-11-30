def load_weights_only(model, weights_path):
    if weights_path.endswith('.keras'):
        from keras.models import load_model
        loaded_model = load_model(weights_path)
        model.set_weights(loaded_model.get_weights())
    else:
        model.load_weights(weights_path, skip_mismatch=True, by_name=True)
    return model


from pyexpat import model
import warnings
import os

from dnn.dnn1 import X_test

def load_weights_no_warning(model, weights_path):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Skipping variable loading for optimizer')
        model.load_weights(weights_path)
    return model


def build_model_matching_saved_state(weights_path):
    from keras.models import load_model

    model = load_model(weights_path)

    return model


def modified_dnn3test_loading():
    import warnings

    score = []
    name = []

    for file in os.listdir("kddresults/dnn3layer/"):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Skipping variable loading for optimizer')
            model.load_weights("kddresults/dnn3layer/" + file)

        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()


def load_weights_silent(model, path):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='keras.src.saving.saving_lib')
        model.load_weights(path)
    return model
