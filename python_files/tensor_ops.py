"""
This file is intended as a library of tensor operations and vectorized
operations written in PySEAL. 

"""
#%%
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
    """Multiply componentwise cipher and plain along cipher's last dimension
    cipher.shape[-1]= plain.shape
    """
    def __init__(self, context):
        self.evaluator = Evaluator(context)
    def __call__(self, cipher, plain):
        """In order to not affect cipher for other calculations, make copy"""
        cipher = dc(cipher)
        if cipher.shape[-1] != plain.shape[0]:
            raise ValueError("The cipher shape and plain shape don't match.")
        for indices in np.ndindex(cipher.shape[:-1]):
            for i in range(plain.shape[0]):
                self.evaluator.multiply_plain(cipher[indices][i],plain[i])
        return cipher
        
class vec_add_many(object):
    """Given an array, this adds all the elements along last dimension """
    def __init__(self, context):
        self.add = Evaluator(context).add_many
    def __call__(self, arr):
        shape = arr.shape[:-1]
        size = int(np.prod(shape))
        ciphers = np.array([Ciphertext() for i in range(size)])
        ciphers = ciphers.reshape(shape)
        for indices in np.ndindex(shape):
            self.add(arr[indices].tolist(),ciphers[indices])
        return ciphers

class vec_add_plain(object):
    """Add bias vector to features. 
    bias: vector.shape = n,m Should be an ndarray of Plaintext()
    matrix: matrix.shape = (batch,n,m). Should be an array of Ciphertext()
    """
    def __init__(self, context):
        self.add = Evaluator(context).add_plain
    def __call__(self, matrix, bias):
        sh = matrix.shape
        if sh[1:] != bias.shape:
            raise ValueError("Incompatible matrix and bias shapes. Make sure matrix tensor is 3D.")
        else:
            shape = matrix[0].shape
            for i in range(sh[0]):
                for indices in np.ndindex(shape): 
                    self.add(matrix[i][indices],bias[indices])
        return matrix.reshape(matrix.shape)

class cipher_dot_plain(object):
    def __init__(self, context):
        self.context = context
        self.multiply = vec_plain_multiply(context)
        self.add = vec_add_many(context)
    def __call__(self,cipher, plain):
        m = self.multiply(cipher, plain)
        out = self.add(m)
        return out

class ciphermatrixprod():
    """Matrix multiplication which allows for batching in 0th dim of first input."""
    def __init__(self,context):
        self.dot = cipher_dot_plain(context)
    def __call__(self,x,y):
        b = x.shape[0]
        d = y.shape[-1]
        if x.shape[-1] != y.shape[0]:
            raise ValueError("Mismatch of shapes")
        for k in range(b):
            copies = [dc(x[np.newaxis,k,...]) for i in range(d)]
            arr = np.array([self.dot(copies[j],y[:,j]) for j in range(d)]).T
            if k == 0:
                out = arr
            else: 
                out = np.concatenate([out, arr], axis=0)
        return np.array(out)

class PlainLinear(object):
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

class Pow(object):
    """Raises x to the power n for n >= 1."""
    def __init__(self, context, keygen):
        self.mult = Evaluator(context).multiply
        self.relinear = vec_relinearize(context, keygen)
    def __call__(self, x, n):
        """For polynomials with multiple degrees, we need to not affect x. 
        This is in reference to mult happening in place or passed to a new
        ciphertext; we need the latter.
        x: array of ciphers
        n: degree int >=1 
        """
        if n == 1:
            return dc(x)
        #xc copy x, mc copy for multiplying
        xc = dc(x)
        mc = dc(x)
        shape = x.shape
        for i in range(n-1):
            for indices in np.ndindex(shape):
                self.mult(xc[indices], mc[indices])
                self.relinear.relinearize(xc)
                #st.write(self.relinear.size(xc))
        return xc

class vec_relinearize():
    def __init__(self, context, keygen):
        self.relinear = Evaluator(context).relinearize
        self.ev_keys = EvaluationKeys()
        keygen.generate_evaluation_keys(30, self.ev_keys)
    def relinearize(self,x):
        shape = x.shape
        for indices in np.ndindex(shape):
            self.relinear(x[indices], self.ev_keys)
        

    def size(self,x):
        shape = x.shape
        sizes = np.empty(shape)
        for indices in np.ndindex(shape):
            sizes[indices] = x[indices].size()
        return sizes

class Poly(object):
    """Compute poly(x) where x is a polynomial with no constant term, and 
    x is an np array of encrypted values. Specifically, instantiating creates
    a polynomial with unknown coefficients. In __call__, coeff are specified.
    This is for nn where all the layers have the same type of polynomial 
    activation functions with trainable coefficients."""
    def __init__(self, context, keygen, coeff, degrees):
        self.pow = Pow(context, keygen)
        self.mul_plain = vec_plain_multiply(context)
        self.add_many = vec_add_many(context)
        self.degrees = degrees
        self.coeff = coeff
    def __call__(self, x):
        powers = np.stack([self.pow(x,d) for d in self.degrees],axis = -1)
        poly = self.mul_plain(powers, self.coeff)
        poly = self.add_many(poly)
        return poly

class vec_noise_budget(object):
    """For inspecting various measures of the noise budget for elements of an array."""
    def __init__(self,decryptor,arr):
        self.shape = arr.shape
        farr = arr.flatten()
        self.__vec_budget = np.vectorize(decryptor.invariant_noise_budget)(farr)
    @property
    def mean(self):
        return np.mean(self.__vec_budget)
    @property
    def min(self):
        return np.min(self.__vec_budget)
    @property
    def budget(self):
        return self.__vec_budget.reshape(self.shape)
    
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

#DELETE ME
def scratchwork():
    """"Pyseal scratch"""
    parms = EncryptionParameters()
    parms.set_poly_modulus("1x^8192 + 1")
    parms.set_coeff_modulus(seal.coeff_modulus_128(8192))
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

    arr1 = np.array([i+1 for i in range(24)]).reshape((2,3,4))
    arr2 = np.array([i+1 for i in range(20)]).reshape((4,5))
    arr3 = np.array([i+2 for i in range(2)])

    code = vec_encoder(encoder)(arr1)
    cipher = vec_encryptor(public_key,context)(code)
    plainy = vec_encoder(encoder)(arr3)
    #bias = vec_encoder(encoder)(arr3)
    out = Poly(context,[1,2])(cipher,plainy)
    #out = vec_plain_multiply(context)(cipher,plainy)
    #out = Pow(context)(cipher,1)
    Out = vec_decryptor(secret_key,context)(out)
    OUT= vec_decoder(encoder)(Out)
    print(OUT)
    print(vec_noise_budget(decryptor,out).budget)



#%%
import numpy as np 
arr1 = np.array([[i+1 for i in range(24)]]).reshape((6,4))
arr1[np.newaxis,0].shape

#%%
