import streamlit as st
import os
from pathlib import Path
import yaml
from server.train_models.nn_train import main as train_main
import client.request_predictions as client
import server.serve_model as serve

DIRECTORY = Path(os.path.realpath(__file__)).parent

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
        model = self.modelselect()
        data = self.dataselect()
        if model != "Select" and data != "Select":
            self.setencryptionparams()
    def setencryptionparams(self):

        security_level = self.centerselectbox("Select security level: ", [128,192])
        
        poly_modulus_pwr = self.centerselectbox("Select polynomial modulus: ", [2**(i+10) for i in range(6)])
        plain_modulus = self.centerselectbox("Select the plaintext modulus", [i+8 for i in range(8)])
        if st.checkbox("Advanced settings."):
            st.write("This setting will override the security level settings. It's not suggested to use this setting. \
                You should enter the product of primes each of which is congruent to 1 modulo 2**(polynomial modulus).")
            coeff_modulus = st.text_input("Enter a coefficient modulus.")
        pass
    def runinference(self):
        pass

def Test():
    pass

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
                continuetrain = False
            if newold == "Old":
                modelloaded = self.old()
                continuetrain = True
            if modelloaded:
                self.training(continuetrain)

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
                return True

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
                return True
            
    def training(self, continuetrain = False):
        st.header("Training...")
        train_main(modeldir = self.modelname, datadir = self.datadir, continuetrain = continuetrain)



EncryptedInference()