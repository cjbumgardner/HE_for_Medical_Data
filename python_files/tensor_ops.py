"""
This file is intended as a library of tensor operations and vectorized
operations written in PySEAL. 

"""

import streamlit as st
import seal 
import numpy as np
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


class vec_encoder(object):
    def __init__(self, encoder):
        self.vec_enco = np.vectorize(encoder.encode)
    def __call__(self,arr):
        empty = np.empty(arr.shape)
        empty = self.vec_enco(arr)
        return empty


class vec_encryptor(object):
    def __init__(self,public_key, context):
        self.vec_encryptor = np.vectorize(Encryptor(context, public_key).encrypt)
        self.parms = context.parms()
    def __call__(self, arr):
        size = arr.size
        empty_cipher = [Ciphertext(self.parms) for i in range(size)]
        cipher_arr = np.array(empty_cipher)
        self.vec_encryptor(arr.flatten(),cipher_arr)
        return cipher_arr.reshape(arr.shape)


class vec_decryptor(object):
    def __init__(self,secret_key, context):
        self.vec_decryptor = np.vectorize(Decryptor(context, secret_key).decrypt)
    def __call__(self, arr):
        size = arr.size
        empty_plain = [Plaintext() for i in range(size)]
        plain_arr = np.array(empty_plain)
        self.vec_decryptor(arr.flatten(),plain_arr)
        return plain_arr.reshape(arr.shape)


class vec_decoder(object):
    def __init__(self, encoder):
        self.vec_deco = np.vectorize(encoder.decode)
    def __call__(self,arr):
        empty = np.empty(arr.shape)
        empty = self.vec_deco(arr)
        return empty

class vec_plain_multiply(object):
    """multiply (cipher, plain)
    cipher.shape= (batch, plain.shape)
    """
    def __init__(self, context, decryptor = None):
        self.context = context
        evaluator = Evaluator(context)
        if decryptor != None:
            self.decryptor = decryptor
        self.vec_plain_muli = np.vectorize(evaluator.multiply_plain)
        self.parms = context.parms()
    def __call__(self, cipher, plain):
        if cipher.shape[1:] != plain.shape[1:]:
            raise ValueError("The cipher shape and plain shape don't match.")
        for i in range(cipher.shape[0]):
            self.vec_plain_muli(cipher[i,:],plain)
        return cipher

class vec_add_many(object):
    """Given a batch of arrays, this adds all the elements of each array """
    def __init__(self, context, decryptor = None):
        self.evaluator = Evaluator(context)
    def __call__(self, arr):
        size = arr.shape[0]
        cipher = [Ciphertext() for i in range(size)]
        for i in range(size):
            self.evaluator.add_many(list(arr[i,:].flatten()),cipher[i])
            cipher = np.array(cipher)
        return cipher.reshape((arr.shape[0],1))

class cipher_dot_plain(object):
    def __init__(self, context):
        self.context = context
        self.multiply = vec_plain_multiply(context)
        self.add = vec_add_many(context)
    def __call__(self,cipher, plain):
        m = self.multiply(cipher, plain)
        return self.add(m)

def print_parameters(context):
    print("/ Encryption parameters:")
    print("| poly_modulus: " + context.poly_modulus().to_string())
    print("| coeff_modulus_size: " + (str)(context.total_coeff_modulus().significant_bit_count()) + " bits")
    print("| plain_modulus: " + (str)(context.plain_modulus().value()))
    print("| noise_standard_deviation: " + (str)(context.noise_standard_deviation()))
    st.write("/ Encryption parameters:")
    st.write("| poly_modulus: " + context.poly_modulus().to_string())
    st.write("| coeff_modulus_size: " + (str)(context.total_coeff_modulus().significant_bit_count()) + " bits")
    st.write("| plain_modulus: " + (str)(context.plain_modulus().value()))
    st.write("| noise_standard_deviation: " + (str)(context.noise_standard_deviation()))
