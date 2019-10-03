"""
This file is intended as a library of tensor operations and vectorized
operations written in PySEAL. 

"""
#TODO multiplication in pyseal won't allow values to be zero. make sure all 0's are epsilon
import streamlit as st
import seal 
import numpy as np
from copy import deepcopy as dc
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
        """encoder: instantiated pyseal encoder object"""
        self.vec_enco = np.vectorize(encoder.encode)
    def __call__(self,arr):
        """arr: ndarray of proper int/float type relative to encoder"""
        empty = np.empty(arr.shape)
        empty = self.vec_enco(arr)
        return empty


class vec_encryptor(object):
    def __init__(self,public_key, context):
        self.vec_encryptor = np.vectorize(Encryptor(context, public_key).encrypt)
        self.parms = context.parms()
    def __call__(self, arr):
        """arr: ndarray of Plaintext() objects."""
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
        self.parms = context.parms()
    def __call__(self, cipher, plain):
        if cipher.shape[-1] != plain.shape[0]:
            raise ValueError("The cipher shape and plain shape don't match.")
        for i in range(cipher.shape[0]):
            for j in range(cipher.shape[1]):
                evaluator.multiply_plain(cipher[i,j],plain[j])
        return cipher
        
class vec_cipher_multiply():
    def __init__(self,context):
        pass
        
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

class vec_add_plain(object):
    """Add bias vector to features. 
    bias: vector.shape = n,m Should be an ndarray of Plaintext()
    matrix: matrix.shape = (batch,n,m). Should be an array of Ciphertext()
    """
    def __init__(self, context):
        evaluator = Evaluator(context)
        self.vec_add = np.vectorize(evaluator.add_plain)
    def __call__(self, matrix, bias):
        sh = matrix.shape
        if sh[1:] != bias.shape:
            raise ValueError("Incompatible matrix and bias shapes. Make sure matrix tensor is 3D.")
        else:
            for i in range(sh[0]):
                self.vec_add(matrix[i,...].flatten(),bias.flatten())
        return matrix.reshape(matrix.shape)

class cipher_dot_plain(object):
    def __init__(self, context):
        self.context = context
        self.multiply = vec_plain_multiply(context)
        self.add = vec_add_many(context)
    def __call__(self,cipher, plain):
        m = self.multiply(cipher, plain)
        return self.add(m)

class ciphermatrixprod():
    """Matrix multiplication which allows for batching in 0th dim of first input."""
    def __init__(self,context):
        self.dot = cipher_dot_plain(context)
    def __call__(self,x,y):
        b = x.shape[0]
        d = y.shape[-1]
        if x.shape[-1] != y.shape[0]:
            raise ValueError("Mismatch of shapes")
        out = None
        for k in range(b):
            copies = [dc(x[k,...]) for i in range(d)]
            arr = np.array([self.dot(copies[j],y[:,j]).squeeze() for j in range(d)]).T
            if out == None:
                out = arr
            else: 
                out = np.stack([out, arr], axis=0)
        return np.array(out)

class PlainLinear():
    """Enacts Ax+b for matrix A and bias b"""
    def __init__(self, context, matrix, bias):
        """
        context: pyseal context object
        matrix: ndarray of properly encoded Plaintext objects
        bias: numpy vector of properly encoded Plaintext objects
        """
        self.mul = ciphermatrixprod(context)
        self.add = vec_add_plain(context)
        self.A = matrix
        self.b = bias
    def __call__(self, x):
        out = self.mul(x, self.A)
        out = self.add(out, self.b)
        return out

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
