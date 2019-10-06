"""
The main server model constructor and evaluator.

This reads data, model choice, and seal context obj from the client; calls the 
model config file (which contains everything needed to set up model); encoded params
to model, sets up model (with params encoded); then evaluates the model; and sends 
the encrypted predictions back to the client.

"""
#TODO fix this for nn, filepaths, and model type paths

import server.seal_functions as sf
from pathlib import Path, PurePath
import streamlit as st

MODELPARMS = Path(__file__).resolve().parent.joinpath("model_params")

MODELS = {"Mortality Risk":
            {"path":"log_reg_mortality",
            "seal_function":sf.linear_reg_svr,
            }
        }   

def build_model_svr(model_keyvalue, inputs, encoder = None, context = None):
    """Builds model from, seal_functions, model params.
        model_keyvalue: key identifying model
        inputs: properly formatted encrypted inputs for model
        encoder: SEAL encoder object
        context: SEAL context object
    """
    modeldict = MODELS[model_keyvalue]
    params_path = MODELPARMS.joinpath(modeldict["path"])
    alias = modeldict["seal_function"]
    try: 
        func = alias(params_path, context=context, encoder=encoder)
    except Exception as e:
        raise ValueError(f"There was a problem with your inputs: {e}")
    return func.eval(inputs)
