
import streamlit as st
import torch
import pickle
import os
from pathlib import Path
import yaml
import time
from tensor_ops import vec_noise_budget
from server.seal_functions import nn_svr, most_recent_model
from server.train_models.nn_train import main as train_main
from server.train_models.nn_train import fully_conn
from client.request_predictions import encryption_handler
import tensor_ops as tops
import numpy as np 

def sigmoid(linear_pred):
    e = np.exp(-linear_pred)
    sigmoid = 1/(1+e)
    return sigmoid

DIRECTORY = Path(os.path.realpath(__file__)).parent

with open(DIRECTORY.parent/"README.md", "r") as f:
    README = f.read()

def testtrain():
    option = st.selectbox("Test or train a model? \
                (testing gives you the options for encrypted or unencrypted environments)",
                ["Select","Test", "Train"])
    return option

def choosedata():
    pass

class Streamlithelp():
    def __init__(self):
        parentdir = DIRECTORY/"server"
        self.models_dir = parentdir/"model_params"
        self.datas_dir = parentdir/"data"
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
        self.datasetdescription = st.empty()
        self.securityleveldescription = st.empty()
        self.polynomialmodulusdescription = st.empty()
        self.plaintextmodulusdescription = st.empty()
        st.sidebar.header("Select a model and dataset")
        self.storebutton = st.sidebar.empty()
        self.printparameters = st.sidebar.empty()
        self.doneencodingmodel = st.sidebar.empty()
        self.getpolymodels()
        self.modeldir = self.sideselectbox("Choose a model", self.polymodels)
        self.absolutemodeldir = self.models_dir/self.modeldir
        self.datadir = self.dataselect()
        if self.datadir != "Select":
            with open(self.datas_dir/self.datadir/"data_dict.pkl", "rb") as f:
                datadict = pickle.load(f)
            self.data_x = datadict["x_"]
            self.data_y = datadict["y_"] 
            self.npdata_x = self.data_x.to_numpy()
            self.npdata_y = self.data_y.to_numpy()
            self.features = self.npdata_x.shape[1]
            st.write("Data_x",self.data_x)

        if st.sidebar.checkbox("More Dataset Information"):
            self.datasetdescription.markdown("More Dataset Information: Add name of data set as a directory \
                in server/data. Then, make a pkl file of a dictionary of pandas dataframes with keys 'x\_'\
                and 'y\_' for the features and targets respectively. ")
       
        if self.modeldir != "Select" and self.datadir != "Select":
            if self.getencryptionparams():
                if self.setencryptionparams():
                    self.getencodedmodel()
                    if self.encodeencryptdata():
                        
                        self.runinference()
                        
            self.runinferencetorch()


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
        self.security_level = self.sideselectbox("Select security level: ", [128,192])
        if st.sidebar.checkbox("Security Level Description"):
            self.securityleveldescription.markdown("Fill in")
        self.poly_modulus_pwr2 = self.sideselectbox("Polynomial modulus: ", [i+10 for i in range(6)])
        if st.sidebar.checkbox("Polynomial Modulus Information"):
            self.polynomialmodulusdescription.markdown("Polynomial Modulus Information: This is the main feature for determining the size of the encrypted messages. \
                Messages are encrypted as polynomials in the ring of polynomials modulo  x<sup>(2^n)</sup>+1. \
                Here you determine n. Larger n means longer inference times, but it will help with \
                evaluating circuits with larger multiplicative depth. For your model try {} first.", unsafe_allow_html=True)
        self.plain_modulus = self.sideselectbox("Plaintext modulus", [i+8 for i in range(8)])
        if self.plain_modulus != "Select":
            self.plain_modulus = 2**(self.plain_modulus)
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
            self.whole = st.sidebar.text_input("Number of whole significant digits")
            decimal = st.sidebar.text_input("Number of decimal significant digits")
            base = st.sidebar.text_input("Base")
            if self.whole != "" and decimal !="" and base != "":
                try:
                    self.whole = int(self.whole)
                    self.decimal = int(decimal)
                    self.base = int(base)
                    coderselect = True
                except:
                    st.sidebar.warning("Make sure you enter integers only.")
            #elif self.whole == "" and decimal == "" and base == "":
            #   coderselect = True

            st.sidebar.markdown("For now, these settings aren't of great use. Future features will be added. \
                The coeff modulus setting will override the security level settings. It's not suggested \
                to use this setting. If used, you should enter the product of primes each of which is \
                congruent to 1 modulo 2\*(polynomial modulus). Also, you can set the plain modulus to a \
                setting that is is also congruent to 1 modulo 2\*(polynomial modulus). This will be useful\
                when future features are added that allow batching of many plaintexts in to on cipher text\
                for more efficient inference time.")

            coeff_modulus = st.sidebar.text_input("Enter a coefficient modulus")
            if coeff_modulus == "":
                self.coeff_modulus = None
            else: 
                self.coeff_modulus = int(coeff_modulus)

            plain = st.sidebar.text_input("Enter a plaintext modulus")
            if plain != "":
                self.plain_modulus = int(plain)

        else:
            coderselect = True
            self.whole = None
            self.coeff_modulus = None

        if "Select" not in [self.plain_modulus, self.poly_modulus_pwr2]:
            if (self.security_level != "Select" or self.coeff_modulus != None) and coderselect:
                return True
            else:
                return False
        else: 
            return False


    def setencryptionparams(self):
        """
        checkbox = self.storebutton.checkbox("Check to store encryption parameters")
        if checkbox == False:
            self.abovestorebutton.warning("Would you like to store encryption parameters?")
        if checkbox:
        """
        check = False
        if isinstance(self.security_level, int):
            if isinstance(self.poly_modulus_pwr2, int):
                if isinstance(self.plain_modulus, int):
                    check = True
        if check:
            self.handler = encryption_handler(security_level=self.security_level,
                                            poly_modulus_pwr2=self.poly_modulus_pwr2,
                                            coeff_modulus=self.coeff_modulus,
                                            plain_modulus=self.plain_modulus,
                                            )
            st.sidebar.markdown(f"Keygen object address: {self.handler.keygen}")
            tops.print_parameters(self.handler.context,self.printparameters)
            if self.whole != "":
                if self.whole == None:
                    self.handler.set_encoder()
                else:
                    self.handler.set_encoder(whole_sign_digits = self.whole,
                                            decimal_sign_digits = self.decimal,
                                            base = self.base,
                                            )   
        return check

    def getencodedmodel(self):
        #nn_model_path, encoder = None, context = None,  keygen = None
        self.encodedmodel = nn_svr(self.absolutemodeldir, 
                                encoder = self.handler.encoder,
                                context = self.handler.context,
                                keygen = self.handler.keygen,
                                )
        self.doneencodingmodel.success(f"Model {self.modeldir} is now encoded.")


    def encodeencryptdata(self):
        
        numdatapoints = self.npdata_x.shape[0]
        index = self.data_x.index
        st.subheader(f"Choose a range of in the {numdatapoints} samples for inference")
        lower = st.text_input("Input lower end of range")
        upper = st.text_input("Input upper end of range")
        encrypted = False
        if (lower != "") and (upper != ""):
            try:
                lower = int(lower)
                upper = int(upper)
                if (lower>= upper) or lower<0 or upper>numdatapoints:
                    st.error(f"You need to make sure you choose 0<= lower < upper < {numdatapoints}")
                else:
                    self.lower = lower
                    self.upper = upper
            except:
                st.error("Make sure to enter numerical index values from the dataframe.")

        if st.checkbox("Encode and encrypt the data in the range selected"):
            st.write("Encoding and encrypting the data")
            start = time.time()
            self.ciphers = self.handler.encode_encrypt(self.npdata_x[lower:upper,:])
            stop = time.time()
            st.success(f"Finished encrypting {upper-lower} samples in {round(stop-start,4)} seconds!")
            encrypted = True
        return encrypted 


    def runinference(self):
        
        if st.checkbox("Run inference with encoded model on encrypted data"):
            start = time.time()
            with st.spinner("This may take a little bit..."):
                self.encoutput = self.encodedmodel.eval(self.ciphers)
            stop = time.time()
            st.success(f"Finished running encoded model with average of \
                {round((stop-start)/(self.upper-self.lower),4)} seconds/sample!")
            st.write(f"Keygen object address: {self.handler.keygen}")
            noise = self.handler.vec_noise_budget(self.encoutput).budget
            st.write(f"The noise budget for the output array is: {noise}")
            with st.spinner("Now decrypting the data and finishing with sigmoid..."):
                self.unencoutput = self.handler.decrypt_decode(self.encoutput)
                st.write(self.unencoutput)
                self.unencoutput = sigmoid(self.unencoutput)
            st.write(self.unencoutput)

    def runinferencetorch(self):
        testmodel = TestNoStreamlit(self.features, self.absolutemodeldir)
        if st.checkbox("Run inference with unencoded model"):
            start = time.time()
            with st.spinner("Running pytorch model..."):
                self.regoutput = testmodel.eval(self.npdata_x[self.lower:self.upper])
            stop = time.time()
            st.success(f"Finished running encoded model with average of {round((stop-start)/(self.upper-self.lower),4)} seconds/sample!")
            st.write(self.regoutput)


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
choices = ["README","Train Model", "Run model on Encrypted Data", "Test Model"]
action = st.sidebar.selectbox("Train, Test encrypted, Test", choices)
if action == "README":
    st.write(README)
if action == "Train Model":
    Train()
if action == "Run model on Encrypted Data":
    EncryptedInference()
if action == "Test Model":
    pass