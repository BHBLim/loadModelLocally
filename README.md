# Load your model locally!
 A script that loads your custom model locally. You can clone this repo, or use it as reference.

**NOTE**: I have not uploaded the machine learning model to this repository, as Git shouldn't really be used for large objects.

## Setup
To set this up, make sure you have Python 3.7.9 installed - don't use 3.9 because certain libraries aren't available for the version yet. You can get it from here:
https://www.python.org/downloads/release/python-379/ - be careful of your existing python installation! Might be a good idea to uninstall your old one first.


You can use GitHub Desktop to easily clone this - or the command line if you're feeling adventurous.

### Virtual Environment

Open up a terminal or cmd/powershell and type

`python -m venv venv` or `python3 -m venv venv` or possibly `python.exe -m venv venv` if your installation is a little strange.

What this does is it sets up a 'virtual environment' called 'venv' which is stored in the venv folder. This will contain all the libraries, the python executable etc that is specific to this project.

You can then activate it with:

`venv/Scripts/activate`

or `source venv/Scripts/activate` if you're on a Mac.

Now, any `pip install` or `python` commands will be installing into/using the venv folder!

### Installing dependencies

Now you can install your requirements, which I've separated into Windows and Mac - note: I'm using an older version of PyTorch because the newer one is way bigger for some reason, and we don't need the new features.

Now you can run:

`pip install -r requirements.txt`

or `pip install -r requirements-mac.txt` if you're on a Mac.

This will read the requirements file and then install the required dependencies into your venv folder.

## Run it!

Great, now you're ready to load your model! Place your model (named `dogscats.pt`) presumably under a new folder called `models` and then you should be able to run:

`python loadlocalmodel.py`

And **voila** you have now loaded the model and identified that the image at `images/dog.jpg` is a dog! (Hopefully!) You can change `IMAGE_PATH` in `loadlocalmodel.py` to specify a different image. You can even adapt the code to accept command line arguments instead of hard-coding it into the script!