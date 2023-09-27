# Genetic Algorithms

University project, aiming at fitting Holt-Winters prediction model to the real word COVD-19 data using Evolution Algorythm.

# Usage
Put `csv` COVID data from the OWID website (https://ourworldindata.org/coronavirus) into the `data` folder and run the `sym_final.py` script. It will fit the necessary HW model parameters to the provided data and save the model into `models` directory. You can see the provided data, as well as the prediction by running `graphs.py` file.

# Parameters

You can tweak the algorythm parameters via the `parameters.json` file:
* `data_path` is the path for the COVID data
* `test_range` is the size of test set in days
* `sigma_0` is the baseline mutation range
* `mi` is the size of the population (one set of parameters is one speciement)
* `lambda` is the size of the newborn population at each algorythm step (`lambda` > `mi`)
* `c_e` and `c_d` are growth and decline rates
* `k` is the number of generations after which the self-adaptation mechanism is applied
* `predictions` is size of the prediction set in days
* `generations` is the number of generations after which the algorythm terminates
