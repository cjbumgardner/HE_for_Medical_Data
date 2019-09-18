"""

"""
import streamlit as st
import pandas as pd
import seal 
from seal import Ciphertext, \
    Decryptor, \
    Encryptor, \
    EncryptionParameters, \
    Evaluator, \
    IntegerEncoder, \
    KeyGenerator, \
    Plaintext, \
    SEALContext

parms = EncryptionParameters()

parms.set_poly_modulus("1x^2048 + 1")

parms.set_coeff_modulus(seal.coeff_modulus_128(2048))

parms.set_plain_modulus(1 << 8)

context = SEALContext(parms)

def print_parameters(context):
    st.write("/ Encryption parameters:")
    st.write("| poly_modulus: " + context.poly_modulus().to_string())
    
    # Print the size of the true (product) coefficient modulus
    st.write("| coeff_modulus_size: " + (str)(context.total_coeff_modulus().significant_bit_count()) + " bits")

    st.write("| plain_modulus: " + (str)(context.plain_modulus().value()))
    st.write("| noise_standard_deviation: " + (str)(context.noise_standard_deviation()))

print_parameters(context)

encoder = IntegerEncoder(context.plain_modulus())

keygen = KeyGenerator(context)
public_key = keygen.public_key()
secret_key = keygen.secret_key()

st.write(f"secret key: {secret_key}")

