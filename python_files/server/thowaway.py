
import torch
from pathlib import Path, PurePath
import pandas as pd 
import streamlit as st
import pickle
data = Path(__file__).parent/"data"/"mortality_risk"
st.write(data)

datadict = data/"smallerfeatures.pkl"
with open(datadict, 'rb') as f:
    pddict = pickle.load(f)


st.write(pddict)
st.write(pddict.columns)
noids = pddict.drop(columns = ["subject_id"])
x_ = noids.drop(columns = ["expire"])
y_ = noids.expire
st.write(x_)
st.write(y_)
newdict = {"x_": x_, "y_": y_}
with open(data/"data_dict.pkl", "wb") as f:
    pickle.dump(newdict,f)

#train = pddict["train"]
#test = pddict["test"]
#st.write(train.columns)
#train_x = train.drop(columns = "expire")
#train_y = train.expire
#test_x = test.drop(columns = "expire")
#test_y = test.expire
#st.write(test_x)
#st.write(test_y)

#newdict = {"train_x": train_x,"train_y":train_y,"test_x":test_x, "test_y":test_y}


#with open(data/"train_dict.pkl", 'wb') as f:
#    pickle.dump(newdict,f)

def crap():
    """"Pyseal scratch"""
    parms = EncryptionParameters()
    parms.set_poly_modulus("1x^4096 + 1")
    parms.set_coeff_modulus(seal.coeff_modulus_128(4096))
    parms.set_plain_modulus(1 << 8)
    context = SEALContext(parms)
    encoder = FractionalEncoder(context.plain_modulus(),context.poly_modulus(),64,32,3)
    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()
    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)
    value1 = 5.0
    plain1 = encoder.encode(value1)
    value2 = -7.0
    plain2 = encoder.encode(value2)
    evaluator = Evaluator(context)
    encrypted1 = Ciphertext()
    encrypted2 = Ciphertext()
    print("Encrypting plain1: ")
    encryptor.encrypt(plain1, encrypted1)
    print("Done (encrypted1)")
    print("Encrypting plain2: ")
    encryptor.encrypt(plain2, encrypted2)
    print("Done (encrypted2)")

    #%%
    arr1 = np.array([i+1 for i in range(12)]).reshape((1,3,4))
    arr2 = np.array([i+1 for i in range(20)]).reshape((4,5))
    arr3 = np.array([i+1 for i in range(5)])
    code = vec_encoder(encoder)(arr1)
    cipher = vec_encryptor(public_key,context)(code)
    plainy = vec_encoder(encoder)(arr2)
    bias = vec_encoder(encoder)(arr3)
    #%%
    ans= ciphermatrixprod(context)(cipher, plainy)
    Ans = vec_decryptor(secret_key,context)(ans)
    ANS= vec_decoder(encoder)(Ans)
    ANS
    #%%
    lin = Linear(context, plainy, bias)(cipher)
    #%%
    Lin = vec_decryptor(secret_key,context)(lin)
    LIN = vec_decoder(encoder)(Lin)
    LIN
