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

This project also has a Makefile which simplifies some commands (if you have make available).
- `make` or `make run` to run the program
- `make install` to install dependencies from requirements.txt
- `make save-deps` to update requirements.txt with current deps.

In `.env`, the DEBUG flag toggles the development mode. `true` corresponds to Desktop mode, `false` corresponds to Pi mode.

## Running on the Pi
To most efficiently interface with devices on the Pi, install the RPi.GPIO library to use as the GPIO Zero pin factory.

```
pip install RPi.GPIO
```

## Running on Desktop
The project has been configured to work on PCs by using the GPIO Zero MockPinFactory.

After starting in Desktop mode, the program waits for an `f` keypress, which behaves as the action button on the physical hardware. To exit the program, use `Ctrl-C`.