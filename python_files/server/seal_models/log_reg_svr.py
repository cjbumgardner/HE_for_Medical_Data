#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:23:03 2019

@author: christopher
"""

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


def enc_dot(v,w):
