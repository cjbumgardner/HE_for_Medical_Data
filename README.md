# Decipher ML
Utilizing Homomorphic Encryption for Secure ML Inference

This is a project developed in four weeks at Insight's Artifical Intelligence Fellowship program. Its purpose is twofold. One, as a testbed for the possibility of utilizing homomorphic encryption for secure inference in the healthcare predictive analytics industry (see below for more details). Two, it is an experimental tool to quickly build, train, and encode fully connected networks to run on encrypted data. This project wouldn't have been possible in four weeks if not for PySEAL from Lab41. PySEAL is a wrapper for the C++ SEAL encryption library out of Microsoft.

## Utilizing Homomorphic Encryption for Secure ML Models
With the growing field of predictive analytics in the healthcare industry, there is a growing need to utilize ML models in a secure manner to protect healthcare records. Homomorphic encryption is an encryption that allows for arithmetic operations on ciphertexts that decrypts to the same arithmetic operations on the plaintexts. In other words, we can run ML inference on homomorphically encrypted data, and the output is decrypted to a value as if we simply ran inference on the raw data. A google slidedeck for this Insight project can be found [here](https://docs.google.com/presentation/d/15EZNeUMWxDNn39WEwgoHw3fQwY_7OZWdggyzbdaYs-I/edit?usp=sharing]).

Healthcare analytics is a good usecase to understand what and why homomorphic encryption could be/is useful. Although, there are many complexites that arise in the model design and encryption implementation process, so there's some work needed to move this technology along. To name just a couple problems:
- Only polynomial tensor networks can be used due to computational complexites arising from the encryption. In my tests, they seem to offer as good of results compared to, say, networks with ReLU activations.
- To encrypt a number one needs to  encode it as a polynomial, then encrypt the polynomial. The encoding is sensitive to the depth of the multiplication circuits, and as well, the encryption is sensitive to the multiplicative depth for almost orthogonal reasons. Either of these problems going wrong can corrupt encrypted circuit computations. 




## Setup
Clone repository and update python path
```

=======
repo_name = HE_for_Medical_Data
git clone https://github.com/cjbumgardner/$repo_name
cd $repo_name
echo "export $repo_name=${PWD}" >> ~/.bash_profile
echo "export PYTHONPATH=$repo_name/src:${PYTHONPATH}" >> ~/.bash_profile
source ~/.bash_profile
```
