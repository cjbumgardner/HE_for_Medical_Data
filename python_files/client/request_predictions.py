
import seal 
from seal import ChooserEvaluator, \
    Ciphertext, \
    Decryptor, \
    Encryptor, \
    EncryptionParameters, \
    Evaluator, \
    FractionalEncoder, \
    IntegerEncoder, \
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
import streamlit as st
import tensor_ops as tops
import client.postprocess as post
import client.preprocess as prep

class encryption_handler(object):
    """
    Methods:
        set_encoder: sets specified encoder
        encode_encrypt_2D: for encoding list of lists
        decrypt_1D: for decoding list
    Attributes:
        params: seal encrpytion parameters object
        batch: bool to batch operations
        context: seal context object
        secretkey: secret key kept on client side
        publickey: public key for encryption
        encoder: encoder for numerical inputs
    """
    def __init__(self,
                security_level = 128,  #128 or 192 for now
                poly_modulus_pwr2 = 12, # 11 through 15
                coeff_modulus = None,
                batch = False,
                ):
        """
        security level: 128 or 192 
        poly_modulus_pwr2: 11,12,13,14,or 15 poly=x^(2^thisvariable)+1
            will define our polynomial ring by Z[x]/poly. Larger 
            number means more security but longer computations.
        coeff_modulus: default None, If set then security level
            is ignored. This is important to set for batching as 
            it needs to be prime.
        batch: default False, setting to true will design encryption
            scheme to allow parallel predictions
        """
        self.params = EncryptionParameters()
        self.batch = batch
        power = 2 ** poly_modulus_pwr2
        self.params.set_poly_modulus(f"1x^{power} + 1")
        if coeff_modulus != None:
            self.params.set_coeff_modulus(coeff_modulus)
            st.write("Security level is ignored since coeff_modulus was set.")
        else:
            if security_level == 128:
                self.params.set_coeff_modulus(seal.coeff_modulus_128(power))
            if security_level == 192:
                self.params.set_coeff_modulus(seal.coeff_modulus_192(power))
        self.params.set_plain_modulus(1<<8)
        try: 
            self.cont = SEALContext(self.params)
        except Exception as e:
            raise ValueError("There was a problem with your parameters.")
            st.write(f"There was a problem with your parameters: {e}")

        tops.print_parameters(self.context)
        keygen = KeyGenerator(self.context)
        self.secretkey = keygen.secret_key()
        self.publickey = keygen.public_key()

    def set_encoder(self,
                    fractional_encoder = True,
                    whole_sign_digits = 32,
                    decimal_sign_digits = 32,
                    base = 3,
                    ):
        if fractional_encoder == True:
            self.enco = FractionalEncoder(self.cont.plain_modulus(),
                                        self.cont.poly_modulus(), 
                                        whole_sign_digits,
                                        decimal_sign_digits,
                                        base,
                                        )
        else:
            self.enco = IntegerEncoder(self.cont.plain_modulus(),
                                        base,
                                        )
    @property
    def context(self):
        return self.cont

    @property
    def encoder(self):
        return self.enco

    def encode_encrypt(self, x):
        """
        Encodes then encrypts a list x according to scheme.
        x: list of samples with features
        """
        encoded = tops.vec_encoder(self.encoder)(x)
        return tops.vec_encryptor(self.publickey,self.context)(encoded)
    
    def decrypt_decode(self, x):
        """
        Decrypts a list that was encrypted with self.publickey.
        x: list of encrypted values.
        """
        out = tops.vec_decryptor(self.secretkey, self.context)(x)
        return tops.vec_decoder(self.encoder)(out)                
                                        
                                        

class request_receive(object):
    def __init__(self, data_path, model):
        if model == "Mortality Risk":
            #for now data_path is just the data and preproc does nothing
            self.pre_proc = lambda x :x 
            self.post_proc = post.log_reg_sigmoid
            self.data = data_path
        else:
            st.write("We're diligently working on supplying more models.")
    @property
    def process_data(self):
        processed = self.pre_proc(self.data)
        return processed

    def post_process(self, predictions):
        return self.post_proc(predictions)
    