FROM pcej/pyseal
RUN mkdir -p HE_for_Medical_Data/
WORKDIR /HE_for_Medical_Data/
COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PYTHONPATH $PYTHONPATH:/HE_for_Medical_Data/python_files

