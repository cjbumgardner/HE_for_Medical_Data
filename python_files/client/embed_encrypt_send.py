"""
Input:
yml
    model_type: 
    security_level:
    bit_precision: (e.g. int32)
ndarray
    data: 

Output:
ndarray 
    data_out: preprocessed and encrypted data, serialized 
seal pk datatype
    pk: public key 

"""
import streamlit as st

class client_side_processing(object):
    """
    
    """
    def __init__(self,
                data,
                path_to_model
                ):
        self.data = data
        self.model = path_to_model

    
    
st.write("hello")
