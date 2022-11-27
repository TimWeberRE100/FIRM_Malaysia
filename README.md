# FIRM_Malaysia
The FIRM_Malaysia software has the following Python package dependencies:
- argparse
- numpy
- scipy
- datetime
- csv
- multiprocessing

The software uses argparse to parse arguments entered into the main function:

Argument    Default     Recommended Values                          Description
-i          400         maxiter=4000,400                            Maximum iterations for the optimisation
-e          5           per-capita electricity: 5,10, and 20MWh     Amount of electricity consumed per capita per year
-n          PMY_only    PMY_only, PMY_BMY, BMY_only                 Scenario for simulation (peninsular only, peninsular and Borneo with HVDC submarine connection, Borneo only)

To run the software for a given scenario:
1. Enter data into the input files. The Calculations.xlsx workbook is used to determine the cost factors entered into factor.csv
2. Run the Optimisation.py script from the terminal with the relevant command line arguments. For Bash:

    $python3 Optimisation.py [-i] [I] [-e] [E] [-n] [N]
