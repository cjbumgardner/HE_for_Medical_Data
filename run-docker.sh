#!/bin/bash

docker run --name pysealHEML -it -w /HE_for_Medical_Data/ -v "$(pwd)"/data:/data -d -p 127.0.0.1:8501:8501  pcej/pyseal:HEML
docker exec pysealHEML streamlit run  /HE_for_Medical_Data/python_files/interactive_traintestencrypt.py
