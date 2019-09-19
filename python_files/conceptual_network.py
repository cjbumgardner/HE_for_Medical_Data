"""

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

