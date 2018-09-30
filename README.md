# Sentiment-Analysis

## Dependencies
The code has two dependencies which are as follows - 
1.sklearn
2.re

For downloading and installing depeindencies two files are give on for windows and the other one for linux enviroment. They are named `dependencies_win.bat` and `dependencies_lin.sh`
Run the files on your respective system to resolve any dependency issue.


Download the data in the file `aclImdb_v1.tar.gz`<br>
The data can alse be downloaded from the original source<a href ="http://ai.stanford.edu/~amaas/data/sentiment/">here</a>.<br>
**We have to organize the file before using it to train our system.**

## Organising Data

Navigate to the Folder where the file was downloaded and Run the shell script named `preprocess.sh`
**Those who encounter problem in running the script can open the script can run individual commands written in the script OR a preprocessed file named `movie_data.tar.gz` has already been provided.
**_For windows Users it is recommended to download the Second file._**

Move the contents of the Movie_data folder generated to a directory named **_data_** in your working directory.

## Organising Code
You may now download the Jupyter Notebook named `Sentiment analysis of Movie Review.ipynb` and connect it to a python kernel.
The Jupyter Notebook is provided for the fast understanding and change of code.

**OR**

You may download the `.py` file and run it through the terminal.
The terminal may look stuck for a long time - **this is becuse the data manipulation on so many files takes some time and is fairly normal** , the code is working just fine.
When Prompted you may enter your own Review and test the Model on it.


###### About the model

Our model is a Logistic classifier and works on vectorized text(Encoded Text).
The model accuracy is about 88% and can change as change in code is done.
The model uses Logistic Regression Function defined in the sklearn(Sci-Kit Learn) package.
