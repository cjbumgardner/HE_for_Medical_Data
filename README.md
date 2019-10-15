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
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)


## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
=======




>>>>>>> dev-initial_build-201909
