# Disaster Response Pipeline Project

This project creates a webapp allowing a user to input a message and classify it based on certain categories. The data that was used to train this model consists of messages sent during a disaster, so the model predicts whether certain messages fall into specific categories related to a disaster.

## Table of Contents
* [Installation](#installation)
* [Project Motivation](#motivation)
* [Instructions](#instructions)
* [File Descriptions](#descriptions)
* [Licensing, Authors, Acknowledgements](#licensing)

## Installation
The code requires Python versions of 3.* and general libraries available through the Anaconda package. In addition, the nltk package needs to be installed for the program to run successfully. For more details on the required packages, please refer to the files train_classifier.py and process_data.py.

## Project Motivation <a name="motivation"></a>
While a natural or other disaster is happening, millions of messages will be sent via social media and it is very difficult for disaster relief workers to determine which messages are relevant and where help needs to be sent. This webapp creates an easy-to-use interface that runs on a machine learning model which categorizes new messages into different categories and helps workers determine whether and what help is needed.

![Screenshot of the webapp](app/static/img/webapp.PNG)

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

	To run ETL pipeline that cleans data and stores in database `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
	To run ML pipeline that trains classifier and saves `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app: `python run.py`

3. Go to http://0.0.0.0:3001/

## File Descriptions <a name="descriptions"></a>
The files `ETL Pipeline Preparation.ipynb` and `ML Pipeline Preparation.ipynb` are jupyter notebooks and contain code exploring the message dataset and preparations for creating the final pipelines in the files `process_data.py` and `train_classifier.py`. The models folder contains `train_classifier.py` which is comprised of several functions reading in data from a sql database and passing the data through a machine learning pipeline, then saving the model to a pickle file, `classifier.pkl` which is also contained in the models folder. The data folder contains the sql database named `DisasterResponse.db`as well as the ETL pipeline file `process_data.py`and two csv files `disaster_categories.csv`and `disaster_messages.csv` that contain the feature and target label data. Finally, the app folder contains a python file, `run.py` that runs the webapp and a static folder with a screenshot of the webapp and a css stylesheet, `style.css`. In the templates folder, there are the two html files contained that are necessary to run the python webapp, `go.html`and `master.html`, the latter being the index file.

## Licensing, Authors, Acknowledgements <a name="licensing"></a>
The data used for the analysis comes from Figure Eight and has been supplied by [Udacity](https://eu.udacity.com/legal/terms-of-use). Feel free to use the code as you please and use it for other dataset containing messages.
