FROM python:3.8-slim

RUN apt update && apt -y install git 
RUN pip install --upgrade pip && \ 
    pip install --no-cache-dir \
        numpy \
        scipy \
        scikit-learn \
        matplotlib \
        pandas \
        imbalanced-learn \
        openpyxl 
RUN pip install --no-cache-dir \
        git+https://github.com/cursecatcher/skope-rules.git

COPY *.py /src/

#ENTRYPOINT [ "/src/rules.py" ]
