"""
<<<<<<< HEAD

"""
import streamlit as st
import pandas as pd
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

parms = EncryptionParameters()

parms.set_poly_modulus("1x^2048 + 1") 
#must be power of 2 cyclotopic polynonial
#ie we're in a field of characteristic 2

parms.set_coeff_modulus(seal.coeff_modulus_128(2048))
#coefficient modulus larger means larger noise budget yet less secure
#here 128 bit security level (see pyseal examples for others available)
#this is a default 56-bit prime factor giving 128 bit security level
#in VR encryption scheme 

parms.set_plain_modulus(1 << 8)

context = SEALContext(parms)
#this sets some precomputations and does a lot of heavy lifting

def print_parameters(context):
    st.write("/ Encryption parameters:")
    st.write("| poly_modulus: " + context.poly_modulus().to_string())
    
    # Print the size of the true (product) coefficient modulus
    st.write("| coeff_modulus_size: " + (str)(context.total_coeff_modulus().significant_bit_count()) + " bits")

    st.write("| plain_modulus: " + (str)(context.plain_modulus().value()))
    st.write("| noise_standard_deviation: " + (str)(context.noise_standard_deviation()))

print_parameters(context)

encoder = IntegerEncoder(context.plain_modulus())
#this is for encoding integers in a manner that can be fed into encryptor

keygen = KeyGenerator(context)
public_key = keygen.public_key()
secret_key = keygen.secret_key()

st.write(f"secret key: {secret_key}")

encryptor = Encryptor(context, public_key)

evaluator = Evaluator(context)

decryptor = Decryptor(context, secret_key)

encrypted1 = Ciphertext()
encrypted2 = Ciphertext()

=======
This is a file to run a conceptual client/server call and response. 
"""
import streamlit as st
import client.request_predictions as client
import server.serve_model as serve
import numpy as np
import pandas as pd
import pickle 
from pathlib import Path
import time

datapath = Path(__file__).parent/"server"/"data"/"mortality_risk"/"smallerfeatures.pkl"

with open(datapath, "rb") as f:
    datadf = pickle.load(f)

datadf = datadf.drop(columns="expire")
datadf = datadf.set_index("subject_id")

tests = ["Mortality Risk", "Chest X-ray Pneumonia Assessment"]
selections = ["select"]+tests 
st.title("Encrypted Prediction Demo")
test = None
test = st.selectbox("Select diagnosis tool.", 
                    selections,
                    index = 0,
                    )

patientid = st.text_input(label = "Enter patient ID.")
if patientid == "":
    pass
elif int(patientid) not in datadf.index:
    st.write("Patient ID not found.")
else:
    patientdata = datadf.loc[int(patientid)]
    st.write(patientdata)
    patientdatanp = patientdata.to_numpy()
    patientdatanp = np.expand_dims(patientdata,axis=0)
levels = ["select", 128, 192]
security = st.selectbox("Select the security level. Higher is \
    more secure but will slow operations on large data.",
    levels,
    )

if security != "select" and test != "select":
    st.subheader("Preprocessing and setting encryption parameters...")
    encryption_handler = client.encryption_handler()
    data_processer = client.request_receive(patientdatanp,test) 
    data_processed = data_processer.process_data
    st.subheader("Encoding and encrypting...")
    encryption_handler.set_encoder()
   
    encrypted_data = encryption_handler.encode_encrypt(data_processed)
    st.write(encrypted_data)
    start = time.time()
#this expects 2D array. Change if you want to batch.
    
    st.subheader("Sending to server...")
    
    st.subheader("Accessing model and encoding parameters \
        according to encryption scheme.")
    encoder = encryption_handler.encoder
    context = encryption_handler.context
    encr_out = serve.build_model_svr(test, 
                                    encrypted_data, 
                                    encoder=encoder, 
                                    context=context,
                                    )
    st.subheader("Received encrypted output.")
    st.write(encr_out)
    st.subheader("Decrypting and decoding...")
    dec_out = encryption_handler.decrypt_decode(encr_out)
    st.write("Transforming data for final output.")
    out = data_processer.post_process(dec_out)
    stop = time.time()
    st.write(f"Elapsed time for prediction: {round(stop-start,4)} seconds")
    ser = pd.Series(out.squeeze(),index= ["Mortality_Risk"])
    out = ser.append(patientdata).rename(patientdata.name)
    st.write(out)
    print(patientdata.name)
>>>>>>> dev-initial_build-201909
