# Modelling input and assumptions
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from Optimisation import percapita, scenario

Nodel = np.array(['PMY','BMY']) # Array of nodes within the electricity network model. Node for peninsular and Borneo Malaysia
PVl =   np.array(['PMY']*1 + ['BMY']*1) # Array of solar regions at nodes 
resolution = 1

MLoad = np.genfromtxt('Data/electricity{}.csv'.format(percapita), delimiter=',', skip_header=1) # EOLoad(t, j), MW - Electricity demand at each node
TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1) # TSPV(t, i), MW

assets = np.genfromtxt('Data/assets.csv', dtype=None, delimiter=',')[1:, 3:].astype(np.float)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
assets = np.genfromtxt('Data/constraints.csv', dtype=None, delimiter=',')[1:, 3:].astype(np.float)
EHydro, EBio = [assets[:, x] for x in range(assets.shape[1])] # GWh
CBaseload = (0.5 * EHydro + EBio) / 8760 # 24/7, GW
CPeak = CHydro - 0.5 * EHydro / 8760 # GW

inter = 0.05
CDCmax = inter * MLoad.sum() / MLoad.shape[0] / 1000 # 5%: PBMY, MW to GW
DCloss = np.array([1350]) * 0.03 * pow(10, -3)

efficiency = 0.8
factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)

firstyear, finalyear, timestep = (2010, 2019, 1)
if scenario == 'PMY_only':
    coverage = np.array(['PMY'])
    
elif scenario == 'PMY_BMY':
    coverage = np.array(['PMY','BMY'])

elif scenario == 'BMY_only':
    coverage = np.array(['BMY'])

else:
    print("Scenario argument is incorrect. Must be PMY_only, BMY_only, or PMY_BMY.")

MLoad = MLoad[:, np.where(np.in1d(Nodel, coverage)==True)[0]]
TSPV = TSPV[:, np.where(np.in1d(PVl, coverage)==True)[0]]

CHydro, CBio = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (CHydro, CBio)]
EHydro = EHydro[np.where(np.in1d(Nodel,coverage)==True)[0]] # GWh
CBaseload = CBaseload[np.where(np.in1d(Nodel,coverage)==True)[0]] # GW
CPeak = CPeak[np.where(np.in1d(Nodel,coverage)==True)[0]] # GW

intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)

pzones = (TSPV.shape[1])
pidx, sidx = (pzones, pzones + nodes)

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW

manage = 0 # weeks
allowance = MLoad.sum(axis=1).max() * 0.05 * manage * 168 * efficiency # MWh

class Solution:
    """A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)"""

    def __init__(self, x):
        self.x = x
        self.MLoad = MLoad
        self.intervals, self.nodes = (intervals, nodes)
        self.resolution = resolution

        self.CPV = list(x[: pidx]) # CPV(i), GW
        self.GPV = TSPV * np.tile(self.CPV, (intervals, 1)) * pow(10, 3) # GPV(i, t), GW to MW
        
        self.CPHP = list(x[pidx: sidx]) # CPHP(j), GW
        self.CPHS = x[sidx] # S-CPHS(j), GWh
        self.efficiency = efficiency

        self.Nodel, self.PVl = (Nodel, PVl)
        self.scenario = scenario

        self.GBaseload, self.CPeak = (GBaseload, CPeak)
        self.CHydro, self.EHydro = (CHydro, EHydro) # GW, GWh

        self.allowance = allowance

    def __repr__(self):
        """S = Solution(list(np.ones(64))) >> print(S)"""
        return 'Solution({})'.format(self.x)