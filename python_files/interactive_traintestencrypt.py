
import streamlit as st
import torch
import pickle
import os
from pathlib import Path
import yaml
import time
import streamlit_extras
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
    

from tensor_ops import vec_noise_budget
from server.seal_functions import nn_svr, most_recent_model
from server.train_models.nn_train import main as train_main
from server.train_models.nn_train import fully_conn
from client.request_predictions import encryption_handler, encryption_runner
import tensor_ops as tops
import numpy as np 
import pandas as pd


DIRECTORY = Path(os.path.realpath(__file__)).parent
HE_for_Medical_Data = DIRECTORY.parent

with open(DIRECTORY.parent/"README.md", "r") as f:
    README = f.read()


def sigmoid(linear_pred):
    e = np.exp(-linear_pred)
    sigmoid = 1/(1+e)
    return sigmoid



class Quantize(object):
    def __init__(self,decimal = 6, zero = 0, scale = 1):
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


def getencodedmodel(modeldir, encoder, context, keygen, quantizer):
    #nn_model_path, encoder = None, context = None,  keygen = None
    encodedmodel = nn_svr(modeldir, 
                        encoder = encoder,
                        context = context,
                        keygen = keygen,
                        quantize = quantizer
                        )
    return encodedmodel


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


class EncryptedInference(Streamlithelp):
    def __init__(self):
        super().__init__()
        st.header("Run Inference with Encoded Models on Encrypted Data")
        if st.checkbox("Description"):
            st.markdown("The main purpose of this tool to help find/validate encryption settings. Depending \
                on the muliplicative depth of the model chosen, the bit length of the model parameters and data \
                features, and the various encryption settings, the encoded model acting on encrypted data may be \
                overwhelmed by noise. If so, the output will be useless. Here, you can observe runtimes of inferencing \
                on encrypted data and check the output cooresponds to the output of the unencrypted settings.")
        #empty placeholders for streamlit notifications and headers
        self.datasetdescription = st.empty()
        self.securityleveldescription = st.empty()
        self.polynomialmodulusdescription = st.empty()
        self.plaintextmodulusdescription = st.empty()
        st.sidebar.header("Select a model and dataset")
        self.printparameters = st.sidebar.empty()
        self.doneencodingmodel = st.sidebar.empty()
        #list models and data
        self.getpolymodels()
        self.modeldir = self.sideselectbox("Choose a model", self.polymodels)
        self.absolutemodeldir = self.models_dir/self.modeldir
        self.dataencryptednotice = st.sidebar.empty()
        self.datadir = self.dataselect()
        #open data selection
        if self.datadir != "Select":
            with open(self.datas_dir/self.datadir/"data_dict.pkl", "rb") as f:
                datadict = pickle.load(f)
            self.data_x = datadict["x_"]
            self.data_y = datadict["y_"] 
            self.npdata_x = self.data_x.to_numpy()
            self.npdata_y = self.data_y.to_numpy()
            self.features = self.npdata_x.shape[1]
            st.write("Data_x",self.data_x)
        #initialize quantize class object. this can be a part of user settings in the future
        self.quantize = Quantize(decimal = 5, zero = 0, scale = 1)
        #once encryption params are set this will load the encryption handler initialized class
        try:
            _handlerdict = find_file_type(self.absolutemodeldir,"cache_handler.pkl")
            if _handlerdict != None:
                with open(_handlerdict, 'rb') as f:
                    encryptdict = pickle.load(f)
                handler = encryptdict["handler"]
                whole = encryptdict["whole"]
                decimal = encryptdict["decimal"]
                base = encryptdict["base"]
                time = encryptdict["time"]
                self.handler = encryption_runner(handler)
                self.handler.set_encoder(whole_sign_digits = whole,
                                        decimal_sign_digits = decimal,
                                        base = base,
                                        )
                tops.print_parameters(self.handler.context, empty = self.printparameters)
            else:
                raise
        except Exception as e:
            self.handler = None
        #once modeldir is selected this encodes the model
        if self.modeldir!="Select" and not isinstance(self.handler, type(None)):
            self.model = getencodedmodel(self.absolutemodeldir, 
                                        self.handler.encoder, 
                                        self.handler._cont, 
                                        None, # for now there are cache problems that my attempts haven't fixed self.handler.keygen,
                                        self.quantize,
                                        )
            self.doneencodingmodel.success(f"Model {self.modeldir} is encoded.")
            showmodelweights(self.absolutemodeldir,quantizer = self.quantize)
        #once subset of data is selected for inference, this encrypts the data    
        try:
            _encrypteddata = find_file_type(self.absolutemodeldir,"cache_data.pkl")
            if _handlerdict != None:
                with open(_encrypteddata, 'rb') as f:
                    d = pickle.load(f)
                    ciphers = d["ciphers"]
                    plain = d["plain"]
                    self.dataencryptednotice.success("Stored Encrypted Data Ready")
            else:
                raise
        except Exception as e:
           ciphers = None
           plain = None
        #the main streamlit selection/action processes
        if st.sidebar.checkbox("More Dataset Information"):
            self.datasetdescription.markdown("More Dataset Information: Add name of data set as a directory \
                in server/data. Then, make a pkl file of a dictionary of pandas dataframes with keys 'x\_'\
                and 'y\_' for the features and targets respectively. ")
       
        choices = ["1. Set encryption parameters",
                    "2. Encrypt Data",
                    "3. Run Inference",
                    ]
       
        st.sidebar.markdown("<div style='background-color:rgba(150,120,150,0.4)'><b> Come here for what's next </b></div>", unsafe_allow_html=True)
        action = self.sideselectbox("Select actions in order.",choices)
        st.sidebar.markdown("----------------------------------")
        if action == "1. Set encryption parameters":
            rawhandlerdict = self.getencryptionparams()
            if self.modeldir != "Select":
                @streamlit_extras.cache_on_button_press("Set encryption parameters", ignore_hash=True)
                def create_encryption_handler(rawhandlerdict):
                    st.write('rawhandlerdict', rawhandlerdict)
                    handlerdict = self.setencryptionparams(**rawhandlerdict)
                    st.write('handlerdict', handlerdict)
                    return handlerdict
                handlerdict = create_encryption_handler(rawhandlerdict)
                st.write('returned handlerdict', handlerdict)
                raise RuntimeError('Blah')
                # if st.sidebar.button(""):
                    

                    # if rawhandlerdict != None:
                    #     handlerdict = self.setencryptionparams(**rawhandlerdict)
                    #     with open(self.absolutemodeldir/"cache_handler.pkl", 'wb') as f:
                    #         pickle.dump(handlerdict, f)
                    #         st.sidebar.success("Encryption Parameters Set")
                    #         tops.print_parameters(handlerdict["handler"].context,self.printparameters)
                    
        if action == "2. Encrypt Data":
            if self.handler == None:
                st.error("You need to set encryption settings")
            else:
                cipherdict = self.encodeencryptdata()
                if not isinstance(cipherdict["ciphers"], type(None)):
                    try:
                        with open(self.absolutemodeldir/"cache_data.pkl", "wb") as f:
                            pickle.dump(cipherdict,f)
                            st.sidebar.success("Data Encoded")
                    except Exception as e:
                        st.sidebar.error(e)

        if action == "3. Run Inference":
            if st.button("Run inference for both encrypted and unencrypted models"):
                if not isinstance(ciphers,type(None)) and not isinstance(plain,type(None)):
                    self.runinference(ciphers,plain)
                else:
                    st.error("You need to encrypt a few samples")

    def getpolymodels(self):
        allmodels = self.selections(self.models_dir)
        self.polymodels = []
        for model in allmodels:
            path = self.models_dir/model/"configs.yaml"
            with open(path,"r") as f:
                configs = yaml.safe_load(f)
            try:
                if configs["activation"] == "poly":
                    self.polymodels.append(model)
            except:
                pass


    def getencryptionparams(self):
        security_level = self.sideselectbox("Select security level: ", [128,192])
        if st.sidebar.checkbox("Security Level Description"):
            self.securityleveldescription.markdown("Fill in")
        poly_modulus_pwr2 = st.sidebar.selectbox("Polynomial modulus: ", [i+10 for i in range(6)], index = 3)
        if st.sidebar.checkbox("Polynomial Modulus Information"):
            self.polynomialmodulusdescription.markdown("Polynomial Modulus Information: This is the main feature for determining the size of the encrypted messages. \
                Messages are encrypted as polynomials in the ring of polynomials modulo  x<sup>(2^n)</sup>+1. \
                Here you determine n. Larger n means longer inference times, but it will help with \
                evaluating circuits with larger multiplicative depth. For your model try {} first.", unsafe_allow_html=True)
        plain_modulus = st.sidebar.selectbox("Plaintext modulus", [i+8 for i in range(15)], index = 2)
        if plain_modulus != "Select":
            plain_modulus = 2**(plain_modulus)
        if st.sidebar.checkbox("Plaintext Modulus Information"):
            self.plaintextmodulusdescription.markdown("Plaintext Modulus Information: Plaintexts are polynomials (numbers will be encoded as polynomials). Like \
                polynomial modulus, this selection is for the power of 2 chosen to be plaintext size. \
                A reasonable setting to start with for your model is {}.")

        if st.sidebar.checkbox("Advanced settings"):
            coderselect = False
            st.sidebar.markdown("Change default encoder settings: Here you can set the significant digits of your numerical calculations. These \
                must adhere to the max number of significant digits you think will be needed in \
                calculations. You can also change the base of your numerical representation, default is base 3.\
                The purpose of using a lower base is related to accommodating proper decoding \
                depending on the depth of circuit calculations. 3 is the default.")
            whole = st.sidebar.text_input("Number of whole significant digits",64)
            decimal = st.sidebar.text_input("Number of decimal significant digits",32)
            base = st.sidebar.text_input("Base",3)
            if whole != "64" or decimal !="32" or base != "3":
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
            if coeff_modulus == "":
                coeff_modulus = None
            else: 
                coeff_modulus = int(coeff_modulus)

            plain = st.sidebar.text_input("Enter a plaintext modulus")
            if plain != "":
                plain_modulus = int(plain)

        else:
            whole = 64
            decimal = 32
            base = 3
            coeff_modulus = None
        if security_level == "Select":
            return None
        else:
            return {"security_level": security_level,
                    "poly_modulus_pwr2": poly_modulus_pwr2,
                    "coeff_modulus": coeff_modulus,
                    "plain_modulus": plain_modulus,
                    "whole": whole,
                    "decimal": decimal,
                    "base": base,
                    "time":time.time(),
                    }

    def setencryptionparams(self,**kwargs):
        security_level = kwargs["security_level"]
        poly_modulus_pwr2 = kwargs["poly_modulus_pwr2"]
        coeff_modulus = kwargs["coeff_modulus"]
        plain_modulus = kwargs["plain_modulus"]
        whole = kwargs["whole"]
        decimal = kwargs["decimal"]
        base = kwargs["base"]
        time = kwargs["time"]
        st.write(kwargs)
        try:
            handler = encryption_handler(security_level=security_level,
                                            poly_modulus_pwr2=poly_modulus_pwr2,
                                            coeff_modulus=coeff_modulus,
                                            plain_modulus=plain_modulus,
                                            )
            st.sidebar.markdown(f"Context object address: {handler.context}")
        except Exception as e:
            st.sidebar.error(f"There was a problem with your encryption settings: {e}")
        return {"handler": handler, "whole": whole, "decimal": decimal, "base": base, "time":time}
    

    def encodeencryptdata(self):
        ciphers = unencrypted = None
        numdatapoints = self.npdata_x.shape[0]
        index = self.data_x.index
        st.subheader(f"Choose a range of in the {numdatapoints} samples for inference")
        lower = st.text_input("Input lower end of range")
        upper = st.text_input("Input upper end of range")
        if (lower != "") and (upper != ""):
            try:
                lower = int(lower)
                upper = int(upper)
                if (lower>= upper) or lower<0 or upper>numdatapoints:
                    st.error(f"You need to make sure you choose 0<= lower < upper < {numdatapoints}")
            except:
                st.error("Make sure to enter numerical index values from the dataframe.")
        #this encodes and encrypts the data as well as QUANTIZE
        if st.button("Encode and encrypt the data in the range selected"):
            try:
                start = time.time()
                with st.spinner("Encoding and Encrypting Data..."):
                    unencrypted = self.quantize(self.npdata_x[lower:upper,:])
                    ciphers = self.handler.encode_encrypt(unencrypted)
                stop = time.time()
                st.success(f"Finished encrypting {upper-lower} samples in {round(stop-start,4)} seconds!")
            except Exception as e:
                st.error(f"There was a problem encryting the data: {e}")
        return {"ciphers": ciphers, "plain": unencrypted}

    def runinference(self, ciphers, plain):
        index, features = ciphers.shape[0], ciphers.shape[1]
        start = time.time()
        with st.spinner("Running encoded model on encrypted data..."):
            encoutput = self.model.eval(ciphers)
        stop = time.time()
        st.success(f"Finished running encoded model with average of \
            {round((stop-start)/index,4)} seconds/sample!")
        noise = self.handler.vec_noise_budget(encoutput)
        if noise.min == 0:
            st.warning("The computations ran out of noise budget.\
                Other internal features will be added in the future to help. For now, adjust the \
                available encryption settings.")
        with st.spinner("Now decrypting the data and finishing with sigmoid..."):
            unencoutput = self.handler.decrypt_decode(encoutput)
            st.write(unencoutput)
            unencoutput = sigmoid(unencoutput)

        testmodel = TestNoStreamlit(features, self.absolutemodeldir)
        start = time.time()
        with st.spinner("Running pytorch model..."):
            regoutput = testmodel.eval(plain)
        stop = time.time()
        st.success(f"Finished running encoded model with average of {round((stop-start)/index,4)} seconds/sample!")
        outstacked = np.concatenate([noise.budget,unencoutput, regoutput], axis=1)

        st.write(pd.DataFrame(outstacked, columns=["Noise budget left", "Encrypted Model", "Unencrypted Model"]))


class TestNoStreamlit():
    def __init__(self, input_size, modeldir):
        with open(modeldir/"configs.yaml", 'r') as f:
            configs = yaml.safe_load(f)
        layers = configs["layers"]
        activation = configs["activation"]
        degrees = configs["degrees"]
        self.testmodel = fully_conn(input_size, layers, activation, degrees=degrees)
        list_of_paths = modeldir.glob("*")
        paths = sorted(list_of_paths, key=lambda p: p.stat().st_ctime)
        paths.reverse()
        for path in paths:
            if path.name[0:5] == "model":
                latest_path = path
                break
        checkpoint = torch.load(latest_path)
        model_state = checkpoint["model_state_dict"]
        self.testmodel.load_state_dict(model_state)
    def eval(self, x):
        x = torch.Tensor(x)
        return self.testmodel.predict(x).detach().numpy()
        

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


st.sidebar.header("Choose an Action")
choices = ["README","Train Model", "Run model on Encrypted Data"]
action = st.sidebar.selectbox("Train, Test encrypted", choices)
st.sidebar.markdown("------------------")
if action == "README":
    st.write(README)
if action == "Train Model":
    Train()
if action == "Run model on Encrypted Data":
    EncryptedInference()


