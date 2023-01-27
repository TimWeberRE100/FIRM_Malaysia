#!/bin/bash

python3 Optimisation.py -e 5 -n PMY_only
python3 Optimisation.py -e 10 -n PMY_only
python3 Optimisation.py -e 20 -n PMY_only
python3 Optimisation.py -e 5 -n BMY_only
python3 Optimisation.py -e 10 -n BMY_only
python3 Optimisation.py -e 20 -n BMY_only
python3 Optimisation.py -e 5 -n PMY_BMY
python3 Optimisation.py -e 10 -n PMY_BMY
python3 Optimisation.py -e 20 -n PMY_BMY