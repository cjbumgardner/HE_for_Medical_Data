"""
This is a file to run a conceptual client/server call and response. 
"""
import streamlit as st
import client.request_predictions as client
import server.serve_model as serve
import numpy as np

tests = ["Mortality Risk", "Chest X-ray Pneumonia Assessment"]
selections = ["select"]+tests 
st.title("Encrypted Prediction Demo")
test = None
test = st.selectbox("Select diagnosis tool.", 
                    selections,
                    index = 0,
                    )
patient = None
patient = st.text_input("Enter patient id or enter PASS for synthetic data.")

if patient == "PASS":
    data = 2*np.random.randn(2,50)+5
    st.write(f"Running {test} for sample:")
    st.dataframe(data)

levels = ["select", 128, 192]
security = st.selectbox("Select the security level. Higher is \
    more secure but will slow operations on large data.",
    levels,
    )

if security != "select" and test != "select":
    st.write("Preprocessing and setting encryption parameters...")
    encryption_handler = client.encryption_handler()
    data_processer = client.request_receive(data,test) 
    data_processed = data_processer.process_data
    st.write("Encoding and encrypting...")
    encryption_handler.set_encoder()
#this expects 2D array. Change if you want to batch.
    encrypted_data = encryption_handler.encode_encrypt(data_processed)
    st.write("Sending to server...")
    st.write(encrypted_data)
    st.write("Accessing model and encoding parameters \
        according to encryption scheme.")
    encoder = encryption_handler.encoder
    context = encryption_handler.context
    encr_out = serve.build_model_svr(test, 
                                    encrypted_data, 
                                    encoder=encoder, 
                                    context=context,
                                    )
    st.write("Received encrypted output.")
    st.write(encr_out)
    st.write("Decrypting and decoding...")
    dec_out = encryption_handler.decrypt_decode(encr_out)
    st.write("Transforming data for final output...")
    out = data_processer.post_process(dec_out)
    st.write(out)
