# Decipher ML
Utilizing Homomorphic Encryption for Secure ML Inference

## Homomorphic Encryption
With the growing field of predictive analytics in the healthcare industry, there is a growing need to utilize ML models in a secure manner to protect healthcare records. Homomorphic encryption is an encryption that allows for arithmetic operations on ciphertexts that decrypts to the same arithmetic operations on the plaintexts. In other words, we can run ML inference on homomorphically encrypted data, and the output is decrypted to a value as if we simply ran inference on the raw data. A google slidedeck for this Insight project can be found [here](https://docs.google.com/presentation/d/15EZNeUMWxDNn39WEwgoHw3fQwY_7OZWdggyzbdaYs-I/edit?usp=sharing]).

## Setup
Clone repository and update python path
```

=======
repo_name = HE_for_Medical_Data
git clone https://github.com/cjbumgardner/$repo_name
>>>>>>> dev-initial_build-201909
cd $repo_name
echo "export $repo_name=${PWD}" >> ~/.bash_profile
echo "export PYTHONPATH=$repo_name/src:${PYTHONPATH}" >> ~/.bash_profile
source ~/.bash_profile
```
