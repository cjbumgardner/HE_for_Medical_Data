# -*- coding: utf-8 -*-

import streamlit as st
import torch
import pickle
import os
from pathlib import Path
import yaml
import time
import streamlit_extras
from tensor_ops import vec_noise_budget
from server.seal_functions import nn_svr, most_recent_model
from server.train_models.nn_train import main as train_main
from server.train_models.nn_train import fully_conn
from client.request_predictions import encryption_handler
import tensor_ops as tops
import numpy as np 
import pandas as pd
from seal import Ciphertext, \
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
import collections


DIRECTORY = Path(os.path.realpath(__file__)).parent
HE_for_Medical_Data = DIRECTORY.parent
parentdir = DIRECTORY/"server"
MODELS_DIR = parentdir/"model_params"
DATAS_DIR = HE_for_Medical_Data/"data"

def selections(dir):
    #returns list of directories in dir
    dirs = list(os.walk(dir))[0][1]
    options = [x for x in dirs if x[0]!="."]
    return options

def select_dataset():
    #select a dataset from DATAS_DIR returns path to dataset directory
    datasets = selections(DATAS_DIR)
    datasets.insert(0,"Select")
    dataset = st.sidebar.selectbox("""Select a dataset""", datasets)
    if dataset != "Select":
        return DATAS_DIR/dataset
    return dataset
    

def select_poly_model():
    #select a polynomial model from MODELS_DIR returns path to model directory
    allmodels = selections(MODELS_DIR)
    polymodels = ["Select"]
    for model in allmodels:
        path = MODELS_DIR/model/"configs.yaml"
        with open(path,"r") as f:
            configs = yaml.safe_load(f)
        try:
            if configs["activation"] == "poly":
                polymodels.append(model)
        except:
            pass
    model = st.sidebar.selectbox("""Select a model""", polymodels)
    if model != "Select":
        return MODELS_DIR/model
    return model

def sigmoid(linear_pred):
    e = np.exp(-linear_pred)
    sigmoid = 1/(1+e)
    return sigmoid

class Quantize(object):
    def __init__(self,decimal = 5, zero = 0, scale = 1):
        self.fix = tops.Round(decimal = decimal, zero = zero, scale = scale)
    def __call__(self,x):
        return self.fix(x)

def find_file_type(dir_to_files, filename):
    if isinstance(dir_to_files,str):
        dir_to_files = Path(dir_to_files)
    list_of_paths = dir_to_files.glob("*.pkl")
    paths = sorted(list_of_paths, key=lambda p: p.stat().st_ctime)
    paths.reverse()
    latest_path = None
    for path in paths:
        if path.name == filename:
            latest_path = path
            break
    return latest_path

def showmodelweights(modeldir,quantizer= (lambda x: x)):
    modeldict = most_recent_model(modeldir)
    modeldict = modeldir/modeldict
    weightdict = torch.load(modeldict)["model_state_dict"]
    if st.checkbox("Show model weights"):
        for k,v in weightdict.items():
            st.markdown(k)
            st.dataframe(quantizer(v.detach().numpy()))

def view_readme():
    with open(DIRECTORY.parent/"README.md", "r") as f:
        readme = f.read()
    st.write(readme)


def getencryptionparams():
        security_level = st.sidebar.radio("Select security level: ", [128,192])
        if st.sidebar.checkbox("Security Level Description"):
            st.markdown("This option will select a polynomial modulus that yields an encryption security level of \
                the selected option.")
        poly_modulus_pwr2 = st.sidebar.selectbox("Polynomial modulus", [i+10 for i in range(6)], index = 3)
        if st.sidebar.checkbox("Polynomial Modulus Information"):
            st.markdown("Polynomial Modulus Information: This is the main feature for determining the size of the encrypted messages. \
                Messages are encrypted as polynomials in the ring of polynomials modulo  x<sup>(2^n)</sup>+1. \
                Here you determine n. Larger n means longer inference times, but it will help with \
                evaluating circuits with larger multiplicative depth. For your model try {} first.", unsafe_allow_html=True)
        plain_modulus = st.sidebar.selectbox("Plaintext modulus", [i+8 for i in range(15)], index = 2)
        if plain_modulus != "Select":
            plain_modulus = 2**(plain_modulus)
        if st.sidebar.checkbox("Plaintext Modulus Information"):
            st.markdown("Plaintext Modulus Information: Plaintexts are polynomials (numbers will be encoded as polynomials). Like \
                polynomial modulus, this selection is for the power of 2 chosen to be plaintext size. \
                A reasonable setting to start with for your model is {}.")

        if st.sidebar.checkbox("Advanced settings"):
            st.sidebar.markdown("Change default encoder settings: Here you can set the significant digits of your numerical calculations. These \
                must adhere to the max number of significant digits you think will be needed in \
                calculations. You can also change the base of your numerical representation, default is base 3.\
                The purpose of using a lower base is related to accommodating proper decoding \
                depending on the depth of circuit calculations. 3 is the default.")
            whole = st.sidebar.text_input("Number of whole significant digits",64)
            decimal = st.sidebar.text_input("Number of decimal significant digits",32)
            base = st.sidebar.text_input("Base",3)
            try:
                whole = int(whole)
                decimal = int(decimal)
                base = int(base)
                coderselect = True
            except:
                st.sidebar.warning("Make sure you enter integers only.")

            st.sidebar.markdown("For now, these settings aren't of great use. Future features will be added. \
                The coeff modulus setting will override the security level settings. It's not suggested \
                to use this setting. If used, you should enter the product of primes each of which is \
                congruent to 1 modulo 2\*(polynomial modulus). Also, you can set the plain modulus to a \
                setting that is is also congruent to 1 modulo 2\*(polynomial modulus). This will be useful\
                when future features are added that allow batching of many plaintexts in to on cipher text\
                for more efficient inference time.")

            coeff_modulus = st.sidebar.text_input("Enter a coefficient modulus")
            if coeff_modulus:
                try: 
                    coeff_modulus = int(coeff_modulus)
                except:
                    st.error("You must enter an integer")
            else: 
                coeff_modulus = None
                
            plain = st.sidebar.text_input("Enter a plaintext modulus")
            if plain:
                try: 
                    plain_modulus = int(plain)
                except:
                    st.error("You must enter an integer")
            else: 
                plain = 2**10
        else:
            whole = 64
            decimal = 32
            base = 3
            coeff_modulus = None
        return {"security_level": security_level,
                "poly_modulus_pwr2": poly_modulus_pwr2,
                "coeff_modulus": coeff_modulus,
                "plain_modulus": plain_modulus,
                "whole": whole,
                "decimal": decimal,
                "base": base,
                }

def setencryptionparams(**kwargs):
        security_level = kwargs["security_level"]
        poly_modulus_pwr2 = kwargs["poly_modulus_pwr2"]
        coeff_modulus = kwargs["coeff_modulus"]
        plain_modulus = kwargs["plain_modulus"]
        whole = kwargs["whole"]
        decimal = kwargs["decimal"]
        base = kwargs["base"]
        try:
            handler = encryption_handler(security_level=security_level,
                                            poly_modulus_pwr2=poly_modulus_pwr2,
                                            coeff_modulus=coeff_modulus,
                                            plain_modulus=plain_modulus,
                                            )
            handler.set_encoder(whole_sign_digits=whole, decimal_sign_digits=decimal, base=base)
        except Exception as e:
            st.sidebar.error(f"There was a problem with your encryption settings: {e}")
        return handler


def do_encryption(data_x, data_y, handler, model):
    quantize = Quantize()
    tops.print_parameters(handler.context)
    try:
        start = time.time()
        with st.spinner("Encoding and Encrypting Data..."):
            unencrypted = quantize(data_x)
            ciphers = handler.encode_encrypt(unencrypted)
        stop = time.time()
        st.success(f"Finished encrypting {data_x.shape[0]} samples in {round(stop-start,4)} seconds!")
    except Exception as e:
        st.error(f"There was a problem encryting the data: {e}")
    try: 
        start = time.time()
        encodedmodel = nn_svr(model, 
                            encoder = handler.encoder,
                            context = handler.context,
                            keygen = handler.keygen,
                            quantize = quantize
                            )
        stop = time.time()
        st.success(f"Finished encoding model in {round(stop-start,4)} seconds!")
    except Exception as e:
        st.error(f"There was a problem encoding the model: {e}")
    st.write(f"Norms:{np.linalg.norm(unencrypted,axis=1)}")
    runinference(ciphers, unencrypted, encodedmodel, handler, model, quantize)


def runinference(ciphers, plain, encodedmodel, handler, absolutemodeldir, quantize):
    index, features = ciphers.shape[0], ciphers.shape[1]
    start = time.time()
    with st.spinner("Running encoded model on encrypted data..."):
        encoutput = encodedmodel.eval(ciphers)
    stop = time.time()
    st.success(f"Finished running encoded model with average of \
        {round((stop-start)/index,4)} seconds/sample!")
    noise = handler.vec_noise_budget(encoutput)
    if noise.min == 0:
        st.warning("The computations ran out of noise budget.\
            Other internal features will be added in the future to help. For now, adjust the \
            available encryption settings.")
    with st.spinner("Now decrypting the data and finishing with sigmoid..."):
        unencoutput = handler.decrypt_decode(encoutput)
        unencoutput = sigmoid(unencoutput)

    testmodel = TestNoEncryption(features, absolutemodeldir, quantize)
    start = time.time()
    with st.spinner("Running pytorch model..."):
        regoutput = testmodel.eval(plain)
    stop = time.time()
    st.success(f"Finished running encoded model with average of {round((stop-start)/index,4)} seconds/sample!")
    outstacked = np.concatenate([noise.budget,unencoutput, regoutput], axis=1)
    st.write(pd.DataFrame(outstacked, columns=["Noise budget left", "Encrypted Model", "Unencrypted Model"]))

class TestNoEncryption:
    def __init__(self, input_size, modeldir, quantize = None):
        self.quantize = quantize
        with open(modeldir/"configs.yaml", 'r') as f:
            configs = yaml.safe_load(f)
        layers = configs["layers"]
        activation = configs["activation"]
        degrees = configs["degrees"]
        self.testmodel = fully_conn(input_size, layers, activation, degrees=degrees, quantize=quantize)
        list_of_paths = modeldir.glob("*")
        paths = sorted(list_of_paths, key=lambda p: p.stat().st_ctime)
        paths.reverse()
        for path in paths:
            if path.name[0:5] == "model":
                latest_path = path
                break
        checkpoint = torch.load(latest_path)
        model_state = checkpoint["model_state_dict"]
        model_state = self.round(model_state)
        self.testmodel.load_state_dict(model_state)
    def eval(self, x):
        x = torch.Tensor(x)
        return self.testmodel.predict(x).detach().numpy()
    def round(self,model_state):
        for k in model_state:
            x = model_state[k].numpy()
            x = self.quantize(x)
            model_state[k] = torch.tensor(x)
        return model_state

def get_dataset_for_encryption(dataset_path):
    with open(DATAS_DIR/dataset_path/"data_dict.pkl", "rb") as f:
        datadict = pickle.load(f)
    data_x = datadict["x_"]
    data_y = datadict["y_"]
    return data_x, data_y

def get_data_range(dataset):
    upper = dataset.shape[0]
    left = st.slider("Select a range of data to encrypt", 0, upper, 0)
    right = st.slider("Select upper", left+1, upper, upper)
    return left, right

def build_a_model():
    # Title
    Train()

class Streamlithelp():
    def __init__(self):
        parentdir = DIRECTORY/"server"
        self.models_dir = parentdir/"model_params"
        self.datas_dir = HE_for_Medical_Data/"data" #parentdir/"data"
    def selections(self,dir):
        #get dirs to old models and return
        dirs = list(os.walk(dir))[0][1]
        options = [x for x in dirs if x[0]!="."]
        return options
    def sideselectboxdir(self,message,dir):
        options = self.selections(dir)
        options.insert(0,"Select")
        selection = st.sidebar.selectbox(message, options)
        return selection

    def sideselectbox(self, message, options):
        options.insert(0, "Select") 
        return st.sidebar.selectbox(message, options)

    def centerselectbox(self, message, options):
        options.insert(0,"Select") 
        return st.selectbox(message, options)

    def modelselect(self):
        return self.sideselectboxdir("Choose a model", self.models_dir)
        
    def dataselect(self):
        return self.sideselectboxdir("Select data set", self.datas_dir)


class Train(Streamlithelp):
    def __init__(self):
        super().__init__()
        st.sidebar.header("Train")
        if self.getdatapath():
            newold = st.sidebar.selectbox("Start making a new model and train, or continue training an old model",
                ["Select","New", "Old"],
                )
            modelloaded = False
            if newold == "New":
                modelloaded = self.new()
            if newold == "Old":
                modelloaded = self.old()

    def getdatapath(self):
        self.datadir = self.dataselect()
        if st.sidebar.checkbox("More info"):
            st.sidebar.markdown("For now, the data format must be a pickled dictionary \
                object with keys train_x, train_y, test_x, test_y. If you wish to train \
                on new data, put it in server/data/DATADIRNAME as train_dict.pkl")
        if self.datadir != "Select":
            return True   

    def new(self):
        allowed = False
        st.sidebar.subheader("Choose new perceptron model parameters.")
        modelname = st.sidebar.text_input("Enter a name for a model. \
            Make sure there are no spaces, and it is a valid directory name format.")
        modelname = modelname.replace(" ","")
        if modelname in self.selections(self.models_dir) and modelname != "":
            st.sidebar.warning("Enter a different model name, that one is already taken.")
            modelnameok = False
        else:
            modelnameok = True
            modelname = self.models_dir/modelname
        modelgeometry = st.sidebar.text_input("Input a list of the number of perceptrons for each layer seperated by commas. \
                                    For example, with logistic regression with two outputs, enter 2.")
        modelgeo = list(modelgeometry.split(","))
       
        if modelgeometry != '':
            try:
                modellayers = [int(i) for i in modelgeo if int(i)>0]
                if len(modellayers) == len(modelgeo):
                    allowed = True
                else:
                    allowed = False
                    raise
            except:
                st.sidebar.warning("You need to enter positive integers seperated by commas for perceptron layers.")
                allowed = False
    
        if allowed:
            for layer in range(len(modelgeo)):
                st.sidebar.markdown(f"Layer {layer} will have {modelgeo[layer]} neurons")

        activation = st.sidebar.selectbox("Type of activation function",
                                 ["Select","Polynomial","ReLU"])

        if activation == "Polynomial":
            activation = "poly"
            degrees = [1,2]

        if activation == "ReLU":
            activation = "relu"
            degrees = None
        advanced = st.sidebar.checkbox("Advanced training features")
        if advanced:
            if activation == "poly":
                degrees = st.sidebar.multiselect("Input the degrees to include in polynomial \
                                             activations. E.g, for a polynomial Ax + Bx^2 + Cx^4,\
                                             enter 1,2,4.", 
                                            options = ["1","2","3","4","5","6"], 
                                            default = ["1","2"],
                                            )
                degrees = [int(x) for x in degrees]
            
            lr = st.sidebar.text_input("Enter the learning rate for ADAM optimizer", value = 0.001)
            b = st.sidebar.text_input("Enter the batch size", value = 30)
            n = st.sidebar.text_input("Enter the number of epochs for training", value = 10)
        else:
            lr, b, n = 0.001, 30, 10

        if allowed == True and activation != "Select":
            st.sidebar.markdown("Great that's everything I need")       

        if allowed and modelnameok and activation != "Select":
            if st.sidebar.button("Click: Save model settings & train"):
                configsdict = {"modeltype": "nn",
                            "learning_rate": float(lr),
                            "batch_size": int(b),
                            "num_epochs": int(n),
                            "activation": activation,
                            "layers": modellayers,
                            "degrees": degrees,
                            }
                os.mkdir(modelname)
                configs = modelname/"configs.yaml"
                with open(configs,'w') as f:
                    yaml.dump(configsdict, f)
                
                st.sidebar.markdown(f"Saving model settings to path: {configs}")
                self.modelname = modelname
                self.configsdict = configsdict
                self.training()

    def old(self):
       
        modelname = self.modelselect()
        if modelname != "Select":
            modelname = self.models_dir/modelname
            configs = modelname/"configs.yaml"
            with open(configs,"r") as f:
                configsdict = yaml.safe_load(f)
            placeholder = st.sidebar.empty()
            if st.sidebar.checkbox("Change trainging settings."):
                lr = st.sidebar.text_input("Enter the learning rate for ADAM optimizer", value = 0.001)
                b = st.sidebar.text_input("Enter the batch size", value = 30)
                n = st.sidebar.text_input("Enter the number of epochs for training", value = 10)
                if st.sidebar.button("Click: Update settings"):
                    configsdict["learning_rate"] = float(lr)
                    configsdict["batch_size"] = int(b)
                    configsdict["num_epochs"] = int(n)
                    placeholder.json(configsdict)
                    with open(configs,'w') as f:
                        yaml.dump(configsdict, f)
            placeholder.json(configsdict)
            if st.sidebar.button("Click: Begin training"):
                self.modelname = modelname
                self.configsdict = configsdict
                self.training()

    def training(self):
        checkpoints = most_recent_model(self.modelname)
        if checkpoints == None:
            continuetrain = False
        else:
            continuetrain = True
        st.header("Training...")
        train_main(modeldir = self.modelname, datadir = self.datadir, continuetrain = continuetrain)

def train_model():
    st.write('Training a model')

def encrypt_a_model():
    st.write('# Encrypt a model')
    # horizontal divider
    st.sidebar.markdown('---')

    dataset = select_dataset()
    model = select_poly_model()
    st.sidebar.markdown('---')
    param_dict = getencryptionparams()
    handler =  setencryptionparams(**param_dict)
    if "Select" not in {dataset,model}:
        data_x, data_y = get_dataset_for_encryption(dataset)
        st.write("""### Dataset view""")
        lower, upper = get_data_range(data_x)
        dataviewplaceholder = st.empty()
        dataviewplaceholder.dataframe(data_x[lower: upper])
        if st.button('Encrypt Model and Run Comparitive Inference'):
            npdata_x = data_x[lower: upper].to_numpy()
            npdata_y = data_y[lower: upper].to_numpy()
            do_encryption(npdata_x, npdata_y, handler, model)

def main():
    """Starting point for the Streamlit app."""
    st.sidebar.markdown("# Christopher's Project")
    app_modes = collections.OrderedDict((
        ('View readme', view_readme),
        ('Build a model', build_a_model),
        ('Encrypt a model', encrypt_a_model)
    ))

    app_mode = st.sidebar.radio("Choose a function:", list(app_modes.keys()))
    app_modes[app_mode]()

if __name__ == "__main__":
    main()