"""
This file is intended as a library of tensor operations written in PySEAL. 
In a client server model where all tensor/algebraic operations happen on the 
server side, we assume the models are pretrained and have the following objects
"""
#%%
import streamlit as st
import numpy as np
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
#%%
"""
parms = EncryptionParameters()
parms.set_poly_modulus("1x^2048 + 1")
parms.set_coeff_modulus(seal.coeff_modulus_128(2048))
parms.set_plain_modulus(1 << 8)
context = SEALContext(parms)
"""
def print_parameters(context):
    print("/ Encryption parameters:")
    print("| poly_modulus: " + context.poly_modulus().to_string())
    print("| coeff_modulus_size: " + (str)(context.total_coeff_modulus().significant_bit_count()) + " bits")
    print("| plain_modulus: " + (str)(context.plain_modulus().value()))
    print("| noise_standard_deviation: " + (str)(context.noise_standard_deviation()))
#%%
class dot_prod_seal(object):
    """Dot product for at least one encrypted input and up to one encoded input.
    This is set up specifically for using floating point numbers """
    def __init__(self, context, 
                       int_prec, 
                       frac_prec,
                       plaintext_base,
                       noise_alarm = None):
        """Input: context: SEAL context object with poly, coeff, and plain modulus set.
                  int_prec: """
        self.encoder = FractionalEncoder(context.plain_modulus(), 
                                         context.poly_modulus(), 
                                         int_prec,
                                         frac_prec, 
                                         plaintext_base)



#%%
parms = EncryptionParameters()
parms.set_poly_modulus("1x^2048 + 1")
parms.set_coeff_modulus(seal.coeff_modulus_128(2048))
parms.set_plain_modulus(1 << 8)
context = SEALContext(parms)
print_parameters(context)
#%%

keygen = KeyGenerator(context)
public_key = keygen.public_key()
secret_key = keygen.secret_key()
encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)
encoder = FractionalEncoder(context.plain_modulus(), context.poly_modulus(), 64, 32, 3)

#%%

rational_numbers = [3.1, 4.159, 2.65, 3.5897, 9.3, 2.3, 8.46, 2.64, 3.383, 2.7]
coefficients = [0.1, 0.05, 0.05, 0.2, 0.05, 0.3, 0.1, 0.025, 0.075, 0.05]
encoder = FractionalEncoder(context.plain_modulus(), context.poly_modulus(), 64, 32, 3)
# encoded base 3, 64 digits for whole part, 32 digits for fractional part.
encrypted_rationals = []
rational_numbers_string = "Encoding and encrypting: "
for i in range(10):
    encrypted_rationals.append(Ciphertext(parms))
    encryptor.encrypt(encoder.encode(rational_numbers[i]), encrypted_rationals[i])
    rational_numbers_string += (str)(rational_numbers[i])[:6]
    if i < 9: rational_numbers_string += ", "
print(rational_numbers_string)
encoded_coefficients = []
encoded_coefficients_string = "Encoding plaintext coefficients: "
for i in range(10):
    encoded_coefficients.append(encoder.encode(coefficients[i]))
    encoded_coefficients_string += (str)(coefficients[i])[:6]
    if i < 9: encoded_coefficients_string += ", "
print(encoded_coefficients_string)

div_by_ten = encoder.encode(0.1)
print("Computing products: ")
for i in range(10):
    evaluator.multiply_plain(encrypted_rationals[i], encoded_coefficients[i])
print("Done")
encrypted_result = Ciphertext()
print("Adding up all 10 ciphertexts: ")
evaluator.add_many(encrypted_rationals, encrypted_result)
print("Done")
print("Noise budget in result: " + (str)(decryptor.invariant_noise_budget(encrypted_result)) + " bits")

#%%
