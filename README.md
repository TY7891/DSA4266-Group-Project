## Genomics Project: Prediction of m6A RNA modifications from direct RNA-Seq data
In this project, we developed a machine learning method to identify m6A modification from direct RNA-Seq data.

The purpose of our script is to predict m6A modification by training a model on a labelled dataset with their gene ID and labels depicting whether m6A modification is present. 

In EDA.ipynb file and prelimcode.ipynb, we did exploratory data analysis on it to learn more about the dataset and discovered that there is data imbalance and preliminarily tested if undersampling or oversampling will help to solve the issue.

In baseline models.ipynb, we implemented a baseline model with logistic regression and compared it with a baseline XGBoost model.

In resampling.ipynb, we tested the various sampling methods and discovered that Tomek links worked the best amongst the other sampling methods.

In models.ipynb and models v2.ipynb, we tested a variety of models with and without hyperparameter tuning before deciding on using XGBoost as our model.

In intermediate model.ipynb, we processed the data with our preprocessing steps and tested the model with different combinations of features, which led to the 2 model that we used for the intermediate ranking submission.

In final model.ipynb, we plotted boxplots for the features and included the standard deviation for the signals to add as a feature. We then did the preprocessing steps and trained the model using XGBoost.  

## System requirements
For our test data, we will be using dataset2 given. We would require a medium instance on AWS and python>=3.8 to run the code.

For larger datasets, a larger AWS instance will be required as the memory needed will increase. We recommend using XL and above to run the script if you are to use the script on a bigger dataset.

The packages required to run the script are listed in the requirements.txt

## Interpretation of output
The output will be the same as the intermediate and final ranking submission, where the file will be in csv format and separated by ",". 

The transcript ID, transcript position and score will be recorded in the csv file, where the score is a value between 0 and 1. It represents the probability that there is a m6A modification at that transcript ID and position.

## Arguments passed to the script
The arguments passed onto the script is the input name and the output name. The script takes in a .gz file and a model file, where the model file will be used to predict m6A modification on the .gz file. 

It will then output a csv file with the above description, named after the user's choice.

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
The script takes in a .gz file and the model file as input, and outputs a csv file that shows the transcript id, transcript position and score, separated by ",".



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

