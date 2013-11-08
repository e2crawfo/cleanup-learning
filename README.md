cleanup-learning
================

Python code for learning a cleanup memory with nengo. Currently uses Nengo 1.4.

apt-get install:
mongodb

pip install:
pymongo
hyperopt
dill

Change at top of optimize.py to point to the correct locations.

Create a link from the Nengo distribution to the lib dir of the repo:
ln -s <path-to-repo>/lib <path-to-nengo>/cleanup_lib

Run the database in one terminal
mongod -f mongo.conf

Run the hyperopt script in another:
cd <path-to-repo>
./run <exp-key>

where <exp-key> is unique string which identifies the current run in the database.


