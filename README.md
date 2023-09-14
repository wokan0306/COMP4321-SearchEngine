# COMP4321-SearchEngine
This is the repo to demonstrate my work in COMP4321.

This project is under conda-based environment. It is recommended to use MacOS to test the program. Please follow the command line procedures below step by step.

1.	Please install either anaconda or miniconda. After changing directory to project directory, the user can create a conda environment my_env in the python version=3.9 with the required packages listed in requirements.txt. Remember to activate the environment myenv.
$	 conda create --name my_env --file requirements.txt
$ conda activate my_env

2.	Run the spider program first. It will also call the indexer functions.
$ python spider.py
3.	Run the app.py to host the flask-based web interface.
python app.py
4.	Access the link http://127.0.0.1:5000 to the browser and you may access the search engine now.

I have a 3-minute demo video showcasing my search engine illustration, and I invite you to watch it. Please feel free to take a look:
https://youtube.com/watch?v=1gbIN-yuaRQ&feature=shared
