
import streamlit as st
import pickle
from pathlib import Path
import torch
"""
import seal 
from seal import ChooserEvaluator, \
    Ciphertext, \
    Decryptor, \
    Encryptor, \
    EncryptionParameters, \
    Evaluator, \
    IntegerEncoder, \
    FractionalEncoder, \
    KeyGenerator, \
    MemoryPoolHandle, \
    Plaintext, \
    SEALContext, \
    EvaluationKeys, \
    GaloisKeys, \
    PolyCRTBuilder, \
    ChooserEncoder, \
    ChooserEvaluator, \
    ChooserPoly
"""
server = Path().cwd().parent
model = server.joinpath("model_params",
                        "log_reg_mortality",
                        "model_24-09-2019-08_12_51",
                        )
st.write(model)
with open(model,"rb") as file:
    model_parms = pickle.load(file)
    st.write(model_parms.items())
st.write(torch.load(model))
class eval_seal_model(object):
    def __init__(self, model, context = None):
        pass
