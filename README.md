## Genomics Project: Prediction of m6A RNA modifications from direct RNA-Seq data
In this project, we developed a machine learning method to identify m6A modification from direct RNA-Seq data.










## Running the model on AWS
First, we will have to check the python version to make sure that it is at least 3.8.
```
ls /usr/bin/python*
```
Next, we will install pip for python 3.8.
```
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8
```
Get the required files and test dataset from github.
```
wget https://github.com/TY7891/DSA4266-Group-Project/raw/main/Unix/requirements.txt
wget https://github.com/TY7891/DSA4266-Group-Project/raw/main/Unix/script.py
wget https://github.com/TY7891/DSA4266-Group-Project/raw/main/Unix/final_model.json
wget https://github.com/TY7891/DSA4266-Group-Project/raw/main/Unix/dataset2.json.gz
```
Install the required packages for the script via pip.
```
python3.8 -m pip install -r requirements.txt
```
Run the script with this command.
```
python3.8 script.py --text_path dataset2.json.gz --model_path final_model.json --output_path dataset2.csv
```
## Installation of python3.8
If it is not at least python 3.8, you can install python 3.8 and above with the following code.

Upgrade and update Ubuntu to the latest version. Then install the required packages.
```
sudo apt update && sudo apt upgrade
sudo apt install wget build-essential libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev  
```
If an error occurs for command not found, you can install additional packages to solve it.
```
sudo apt-get install software-properties-common
```
Download and install python3.8 including pip.
```
sudo add-apt-repository ppa:deadsnakes/ppa #download python3.8
sudo apt install python3.8
sudo apt install python3.8-distutils
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8 #get bootstrap 
```

