# Decipher ML
Utilizing Homomorphic Encryption for Secure ML Inference

This is a project developed in four weeks at Insight's Artifical Intelligence Fellowship program. Its purpose is twofold. One, as a testbed for the possibility of utilizing homomorphic encryption for secure inference in the healthcare predictive analytics industry (see below for more details). Two, it is an experimental tool to quickly build, train, and encode fully connected networks to run on encrypted data. This project wouldn't have been possible in four weeks if not for PySEAL from Lab41. PySEAL is a wrapper for the C++ SEAL encryption library out of Microsoft.

## Utilizing Homomorphic Encryption for Secure ML Models
With the growing field of predictive analytics in the healthcare industry, there is a growing need to utilize ML models in a secure manner to protect healthcare records. Homomorphic encryption is an encryption that allows for arithmetic operations on ciphertexts that decrypts to the same arithmetic operations on the plaintexts. In other words, we can run ML inference on homomorphically encrypted data, and the output is decrypted to a value as if we simply ran inference on the raw data. A google slidedeck for this Insight project can be found [here](https://docs.google.com/presentation/d/15EZNeUMWxDNn39WEwgoHw3fQwY_7OZWdggyzbdaYs-I/edit?usp=sharing]).

Healthcare analytics is a good usecase to understand what and why homomorphic encryption could be/is useful. Although, there are many complexites that arise in the model design and encryption implementation process, so there's some work needed to move this technology along. To name just a couple problems:
- Only polynomial tensor networks can be used due to computational complexites arising from the encryption. In my tests, they seem to offer as good of results compared to, say, networks with ReLU activations.
- To encrypt a number, one needs to  encode it as a polynomial, then encrypt the polynomial. The encoding is sensitive to the depth of the multiplication circuits, and as well, the encryption is sensitive to the multiplicative depth for almost orthogonal reasons. Either of these problems going wrong can corrupt encrypted circuit computations. This tool is in early stages of helping test these issues by providing a quick way to change encryption settings and network designs to experiment.


## Future Work
 
- Quantization of the networks and datasets are necessary. Currently, this tool doesn't support this. This would be a major improvement. And, since pytorch has a quantization libary that is fairly new, hopefully a lot of those tools would speed this process up. (Tensorflow has quantization support as well, if not more.)
- Allow for more control over internal 'relinearization' features (this is a feature of the encryption scheme that helps 
reduce ciphertext size after multiplications). Right now, there are simply default settings after every cipher-cipher multiplication. This feature would find settings that balance speed with the accuracy of the network.
- (Much larger piece) Currently (as I'm aware of as of Oct 2019), the SEAL libary doesn't yet implement 'bootstrapping'; a feature to reduce noise in the cipher text with out decrypting. As of now, the encryption scheme used is only partially homomorphic, meaning it has a maximum depth of circuits it can compute homomorphically. To be fully homomorphic (i.e. to be able to compute any depth circuit homomorphically), we need the mentioned 'bootstrapping' feature. There are other HE schemes that use 'modulus switching' to achieve the same effect. This feature would be necessary for deep neural networks, but for now, optimizing partially homomophic encryption with shallow networks for speed and accuracy is both critical and accessible. 





## Setup
Clone repository and update python path
```
=======
repo_name = HE_for_Medical_Data
git clone https://github.com/cjbumgardner/$repo_name
cd $repo_name

```

The simpilest way to interact with this code base is by a Docker container. You need to have docker installed, you can go [here](https://docs.docker.com/install/) for information specific to your OS. 

After the above steps, to build the image: 
```
bash docker-build.sh

```
To run a container and start streamlit app at localhost:8501 in your browser:
```
bash run-docker.sh
```

If you find you used encryption settings and network size too large for your CPU, you'll find you might need to restart 
streamlit.

```
bash restart-streamlit.sh
```

To stop and remove a container:

```
bash exit-docker.sh
```