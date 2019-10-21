"""
This file will be removed in the future.
"""

import server.seal_functions as sf
from pathlib import Path, PurePath
import os
import streamlit as st

MODELPARMS = Path(os.path.realpath(__file__)).parent.joinpath("model_params")

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
