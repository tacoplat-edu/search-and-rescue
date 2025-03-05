# search-and-rescue
MTE 380 Course Project

## Installation
Please use a python virtual env on 3.13.2 to ensure dependency consistency.
Learn more at [Python docs](https://docs.python.org/3/library/venv.html).

In terminal, use the following to install from the current package list.
```
pip install -r requirements.txt
```
and use this to save dependencies after installing them.
```
pip freeze > requirements.txt
```

## Running on the Pi
To most efficiently interface with devices on the Pi, install the RPi.GPIO library to use as the GPIO Zero pin factory.

```
pip install RPi.GPIO
```