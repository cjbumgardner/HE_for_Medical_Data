"""This a file of functions to run on encrypted inputs using PySEAL.
All models need:
__init__ args:
    encoder: to encode parameters properly
    keygen: initialized SEAL key generator object
    context: SEAL context object for knowing the multiplication ring.
    model_dir: path to directory with the model parameters, the model file with 
        the most recent date on it is chosen
__call__ args:
    x: encrypted input encoded with the same encoder from args in __init__

"""
import torch
import tensor_ops as tops
import os
from pathlib import Path
import yaml
import streamlit as st
import numpy as np


def most_recent_model(dir_to_models):
    if isinstance(dir_to_models,str):
        dir_to_models = Path(dir_to_models)
    list_of_paths = dir_to_models.glob("*")
    paths = sorted(list_of_paths, key=lambda p: p.stat().st_ctime)
    paths.reverse()
    for path in paths:
        if path.name[0:5] == "model":
            latest_path = path
            break
    return latest_path

class linear_reg_svr(object):
    """
    Methods:
        eval: evaluates logistic regression on input
    Attributes: 
        weight_v: (list) normalized weight from log_reg training
        enc_weight_v: (list) weights encoded with 
    """
    def __init__(self, log_reg_model_dir, encoder = None, context = None, keygen=None):
        """
        log_reg_model_path: path dir with model weights.
        context: SEAL context object for encryption scheme
        """
        
        try:
            latest_path = most_recent_model(log_reg_model_dir)
            modeldict = torch.load(latest_path)
            weight_v = modeldict["out.weight_v"].numpy()
            weight_g = modeldict["out.weight_g"].numpy()
            bias = modeldict["out.bias"].numpy()
        except Exception as e:
            raise ValueError(f"There was a problem with loading log_reg_model_path: {e}.")
        weight = weight_v*weight_g
        try:
            vencoder = tops.vec_encoder(encoder)
            self.enc_weight = vencoder(weight).squeeze()
            self.enc_bias = vencoder(bias)
        except Exception as e: 
            raise ValueError(f"There was a problem encoding {e}")
        self.dot = tops.cipher_dot_plain(context)
        self.add = tops.vec_add_plain(context)
    def eval(self,x):
        """
        Inputs:
            x: encrypted inputs for single sample
        Returns: encrypted dot product """
        y = self.dot(x, self.enc_weight)
        y = np.expand_dims(y,axis=-1)
        print(y.shape,self.enc_bias.shape)
        y = self.add(y, self.enc_bias)
        
        return y

class nn_svr(object):
    """
    Fully connected NN with poly activations. 
    Properties: 
        context
    """

    def __init__(self, nn_model_path, encoder = None, context = None,  keygen = None):

        coder = tops.vec_encoder(encoder)
        if isinstance(nn_model_path, str):
            nn_model_path = Path(nn_model_path)
        modelstrupath = nn_model_path/"configs"
        recent = most_recent_model(nn_model_path)
        modelparmpath = nn_model_path/recent
        #load model geometry
        cfgs = nn_model_path.joinpath("configs.yaml")
        try:
            with open(cfgs) as f:
                configs = yaml.load(f,Loader = yaml.FullLoader)
        except FileNotFoundError as e:
            raise ValueError("There was a problem finding config.yml.")
        except Exception as e:
            raise ValueError(f"There was an exception: {e}")
        #Get model parameters
        degrees = configs["degrees"]
        layers = configs["layers"]
        l = len(layers)
        #load model parameters
        try:
            modelparm = torch.load(modelparmpath)["model_state_dict"]
        except Exception as e:
            raise ValueError(f"There was a problem with loading log_reg_model_path: {e}.")
        #turn pytorch to numpy tensors
        #due to weight normalization, we need to recombine weight_g weight_v
        #set up pyseal linear op
        self.model = []
        for i in range(l):
            b = modelparm[f"nnet.weightedLinear{i}.bias"].numpy().astype(np.float32)
            wg = modelparm[f"nnet.weightedLinear{i}.weight_g"].numpy().astype(np.float32) 
            wv = modelparm[f"nnet.weightedLinear{i}.weight_v"].numpy().astype(np.float32)
            w = wv * wg
            bcode = coder(b)
            wcode = coder(w).T
            self.model.append(tops.PlainLinear(context, wcode, bcode))
            if i < l-1:
                p = modelparm[f"nnet.poly{i}.coefficients"].numpy()
                pcode = coder(p)
                self.model.append(tops.Poly(context, keygen, pcode, degrees))

    def eval(self,x):
        """
        Inputs:
            x: encrypted inputs for single sample
        Returns: NN model acting on x
         """
        for layer in self.model:
            x = layer(x)
        return x


#PYseal scratch. experiments leave this change dir out. 
def experients():
    import streamlit as st
    import seal 
    import numpy as np
    import pickle
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

    pathmodels = Path(__file__).parent/"model_params"
    st.write(pathmodels)
    modely = pathmodels/"nn_poly13small_mortality"
    st.write("Path to model:", modely)
    pathdata = Path(__file__).parent/"data"/"mortality_risk"/"train_dict.pkl"
    with open(pathdata,"rb") as f:
        data = pickle.load(f)["test"]
    live = data.iloc[5].drop("expire")
    die = data.iloc[35].drop("expire")
    st.write(live)
    st.write(die)
    live = live.to_numpy()
    die = die.to_numpy()
    parms = EncryptionParameters()
    parms.set_poly_modulus("1x^8192 + 1")
    parms.set_coeff_modulus(seal.coeff_modulus_128(4096))
    parms.set_plain_modulus(1 << 10)
    context = SEALContext(parms)
    encoder = FractionalEncoder(context.plain_modulus(),context.poly_modulus(),32,32,3)
    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()
    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)
    evaluator = Evaluator(context)

    vec_coder = tops.vec_encoder(encoder)
    vec_cipher = tops.vec_encryptor(public_key,context)
    live_code = vec_coder(live)
    die_code = vec_coder(die)
    live_cipher = vec_cipher(live_code)
    die_cipher = vec_cipher(die_code)

    batch = np.stack([live_cipher, die_cipher], axis=0)
    st.write("batch shape:",batch.shape)

    network = nn_svr(modely, encoder=encoder, context=context, keygen=keygen)
    out = network.eval(batch)
    st.write(out)
    st.write(tops.vec_noise_budget(decryptor,out).budget)
