"""
download clinc150 from google drive at the first time

You can also download data/ from https://github.com/clinc/oos-eval,

'
    git clone https://github.com/clinc/oos-eval.git
    cd oos-eval
    mv data/ clinc150/
    mv clinc150/ your_workspace/data/clinc150
'

refer to paper: https://www.aclweb.org/anthology/D19-1131/
An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction EMNLP 2019

Author: Tong
Time: 20-02-2021
"""

import requests
import zipfile
import os


def download_file_from_google_drive(id: str, destination_dir: str, file_name: str = "target"):
    URL = "https://docs.google.com/uc?export=download"
    
    # get the path of output file
    if len(file_name.split(".")) == 1:
        file_name = file_name + ".zip"
    out_file = os.path.join(destination_dir, file_name)
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': id}, stream=True)
    token = _get_confirm_token(response)
    
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    _save_response_content(response, out_file)
    
    if zipfile.is_zipfile(out_file):
        # check zip file
        zipped_file = zipfile.ZipFile(out_file)
        # unzip
        zipped_file.extractall(destination_dir)
        # remove zip file
        # os.remove(out_file)

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    
    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

