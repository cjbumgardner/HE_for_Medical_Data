"""
This file is intended as a library of tensor operations and vectorized
operations written in PySEAL. 

"""

import streamlit as st
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

def encode_many(list_, encoder):

    """
    Encodes each element of list_ with encoder.
    list_: list of valid numbers for use with encoder
    encoder: a SEAL encoder object

    Returns: List of encoded inputs
    """
    try:
        list_encoded = list(map(lambda x: encoder.encode(x), list_))
        return list_encoded
    except: 
        raise ValueError("There was an encoding problem.")

def encrypt_many(list_, public_key, context):

    """
    Encrypts list elements.
    list_: list of encoded elements
    public_key: public key made from context 
    context: context object of encryption scheme
    Returns: List of encrypted inputs.
    """
    
    try:
        encryptor = Encryptor(context, public_key)
        parms = context.parms()
        ciphers = []
        for x in list_:
            ciphers.append(Ciphertext(parms))
            encryptor.encrypt(x, ciphers[-1])
        return ciphers
    except: 
        raise ValueError("There was an encrypting problem.")

def decrypt_many(list_, secret_key, context, encoder = None):
    """
    Decrypts a list of ciphers. If encoder is set, then it also decodes.
    list_: list of Ciphertest()
    secret_key: made from context object
    encoder: if supplied it is used to decode 
    """
    try: 
        decryptor = Decryptor(context, secret_key)
        decrypted = []
        for x in list_:
            decrypted.append(Plaintext())
            decryptor.decrypt(x,decrypted[-1])
        if encoder == None:
            return decrypted
        else:
            return [encoder.decode(x) for x in decrypted]
    except:
        raise ValueError("There was a problem either decrypting or decoding.")

def print_parameters(context):
    """
    Prints setup parameters for encrpytion. 
    """
    print("/ Encryption parameters:")
    print("| poly_modulus: " + context.poly_modulus().to_string())
    print("| coeff_modulus_size: " + (str)(context.total_coeff_modulus().significant_bit_count()) + " bits")
    print("| plain_modulus: " + (str)(context.plain_modulus().value()))
    print("| noise_standard_deviation: " + (str)(context.noise_standard_deviation()))

class cipher_dot_plain(object):
    """
    Dot product of a cipher and plain text. This multiplication won't affect noise
    budget, but the addition will.  
    """
    def __init__(self, context, decryptor = None):
        """
        Input: context: SEAL context object with poly, coeff, and plain modulus set.
        """

        self.context = context
        self.evaluator = Evaluator(context)
        self.decryptor = decryptor

    def __call__(self, cipher, plain, batch = False):
        """
        Inputs: 
            cipher: list of encrpyted object(s) of shape n or shape (batch, n)
            plain: list of encoded coefficients of shape n
            batch: bool to turn on batching, default False
        """
        if batch == False:
            pl = len(plain)
            cy = len(cipher)
            if pl != cy:
                raise ValueError(f"Mismatched lengths: cypher: {cy} plaintext: {pl}")
            for i in range(pl):
                self.evaluator.multiply_plain(cipher[i], plain[i]) #alters 1st arg inplace
            encrypted_result = Ciphertext()
            self.evaluator.add_many(cipher, encrypted_result)
            if self.decryptor != None:
                noise = self.decryptor.invariant_noise_budget(encrypted_result)
                print(f"Noise budget after dot product:{noise}")

        else: #TODO finish batching
            crtbuilder = PolyCRTBuilder(context)

        return encrypted_result

 
#%%
parms.set_poly_modulus("1x^4096 + 1")
parms.set_coeff_modulus(seal.coeff_modulus_128(4096))
# Note that 40961 is a prime number and 2*4096 divides 40960.
parms.set_plain_modulus(40961)

context = SEALContext(parms)
keygen = KeyGenerator(context)
public_key = keygen.public_key()
secret_key = keygen.secret_key()
encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)
encoder = FractionalEncoder(context.plain_modulus(), context.poly_modulus(), 64, 32, 3)

cipher = [1.5,2.3,3.97,4.8,10.2]
plain = [1.02, 3.3, -2.4, -8.9, 5.45]

cipher_encode = encode_many(cipher, encoder)
plain_encode = encode_many(plain, encoder)

encrypted = encrypt_many(cipher_encode, public_key, context)

dotted = cipher_dot_plain(context,decryptor)(encrypted,plain_encode)

ans = decrypt_many([dotted],secret_key,context,encoder)[0]
print(ans)

