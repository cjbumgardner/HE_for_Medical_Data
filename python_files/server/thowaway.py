#%%
import torch
from pathlib import Path, PurePath

path = Path("/Users/christopher/datadatdat/Insight/HE_for_Medical_Data")
modelfile = "python_files/server/model_params/log_reg_mortality"

m = path.joinpath(modelfile)
m
#%%
def most_recent_model(dir_to_models):
    list_of_paths = dir_to_models.glob("*")
    paths = sorted(list_of_paths, key=lambda p: p.stat().st_ctime)
    paths.reverse()
    for path in paths:
        if path.name[0:5] == "model":
            latest_path = path
            break
    return latest_path

modeldict = torch.load(most_recent_model(m))


#%%
modeldict.keys()

#%%
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
