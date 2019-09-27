"""This a file of functions to run on encrypted inputs using PySEAL.
They all need:
    encoder: to encode parameters properly
    context: SEAL context object for knowing the multiplication ring.
    parameters: path to parameters for the model

All classes should be of this basic structure:
 
class FUNCTION:
   def __init__(parameters, encoder=None, context=None):
   def eval(input) 
"""

import torch
import tensor_ops as tops 
import streamlit as st

class linear_reg_svr(object):
    """
    Methods:
        eval: evaluates logistic regression on input
    Attributes: 
        weight_v: (list) normalized weight from log_reg training
        enc_weight_v: (list) weights encoded with 
    """
    def __init__(self, log_reg_model_path, encoder = None, context = None):
        """
        log_reg_model_path: path to model weights
        context: SEAL context object for encryption scheme
        """
        
        try:
            model = log_reg_model_path
            model_params_dict = torch.load(model)
            for k,v in model_params_dict.items():
                if k == "out.weight_v":
                    self.weight_v = v.numpy()
            
        except Exception as e:
            raise ValueError(f"There was a problem with loading log_reg_model_path: {e}.")

        self.encoder = encoder
        self.context = context
        try:
            self.enc_weight_v = tops.vec_encoder(encoder)(self.weight_v)
        except Exception as e: 
            raise ValueError(f"There was a problem encoding {e}")
    def eval(self,x):
        """
        Inputs:
            x: encrypted inputs for single sample
        Returns: encrypted dot product """

        y = tops.cipher_dot_plain(self.context)(x,self.enc_weight_v)
        return y
