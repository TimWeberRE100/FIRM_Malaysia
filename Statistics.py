# Load profiles and generation mix data (LPGM) & energy generation, storage and transmission information (GGTA)
# based on x/capacities from Optimisation and flexible from Dispatch
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import *
from Simulation import Reliability
from Network import Transmission

import numpy as np
import datetime as dt

def Debug(solution):
    """Debugging"""

    Load, PV = (solution.MLoad.sum(axis=1), solution.GPV.sum(axis=1))
    Baseload, Peak = (solution.MBaseload.sum(axis=1), solution.MPeak.sum(axis=1))

    Discharge, Charge, Storage = (solution.Discharge, solution.Charge, solution.Storage)
    Deficit, Spillage = (solution.Deficit, solution.Spillage)

    PHS = solution.CPHS * pow(10, 3) # MWh
    efficiency = solution.efficiency

    for i in range(intervals):
        assert abs(Load[i] + Charge[i] + Spillage[i]
                   - PV[i] - Baseload[i] - Peak[i] - Discharge[i] - Deficit[i]) <= 1

        # Discharge, Charge and Storage
        if i==0:
            assert abs(Storage[i] - 0.5 * PHS + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1
        else:
            assert abs(Storage[i] - Storage[i - 1] + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1

        # Capacity: PV, Discharge, Charge and Storage
        try:
            assert np.amax(PV) <= sum(solution.CPV) * pow(10, 3), print(np.amax(PV) - sum(solution.CPV) * pow(10, 3))
            
            assert np.amax(Discharge) <= sum(solution.CPHP) * pow(10, 3), print(np.amax(Discharge) - sum(solution.CPHP) * pow(10, 3))
            assert np.amax(Charge) <= sum(solution.CPHP) * pow(10, 3), print(np.amax(Charge) - sum(solution.CPHP) * pow(10, 3))
            assert np.amax(Storage) <= solution.CPHS * pow(10, 3), print(np.amax(Storage) - sum(solution.CPHS) * pow(10, 3))
        except AssertionError:
            pass

    print('Debugging: everything is ok')

    return True

def LPGM(solution):
    """Load profiles and generation mix data"""

    Debug(solution)

    C = np.stack([solution.MLoad.sum(axis=1),
                  solution.MHydro.sum(axis=1), solution.GPV.sum(axis=1), 
                  solution.Discharge, solution.Deficit, -1 * solution.Spillage, -1 * solution.Charge,
                  solution.Storage, 
                  solution.PBMY])

    C = np.around(C.transpose())

    header = 'Operational demand,' \
             'Hydropower & other renewables,Solar photovoltaics,Pumped hydro energy storage,Energy deficit,Energy spillage,PHES-Charge,' \
             'PHES-Storage,' \
             'PBMY'

    np.savetxt('Results/LPGM_Malaysia_simple{}{}.csv'.format(scenario, percapita), C, fmt='%f', delimiter=',', header=header, comments='')

    print('Load profiles and generation mix is produced.')

    return True

def GGTA(solution):
    """GW, GWh, TWh p.a. and A$/MWh information"""
    
    factor = np.genfromtxt('Data/factor.csv', dtype=None, delimiter=',', encoding=None)
    factor = dict(factor)

    CPV, CPHP, CPHS = (sum(solution.CPV), sum(solution.CPHP), solution.CPHS) # GW, GWh
    CapHydro = (CHydro + CBio).sum() # Hydropower & other resources: GW
    
    GPV, GHydro = map(lambda x: x * pow(10, -6) * resolution / years,
                                              (solution.GPV.sum(), solution.MHydro.sum())) # TWh p.a.
    CFPV = (GPV / CPV / 8.76)

    CostPV = factor['PV'] * CPV # A$b p.a.
    CostHydro = factor['Hydro'] * GHydro # A$b p.a.
    CostPH = factor['PHP'] * CPHP + factor['PHS'] * CPHS - factor['LegPH'] # A$b p.a.
    CostDC = np.array([factor['PBMY']])
    CostDC = (CostDC * solution.CDC).sum() - factor['LegINTC'] # A$b p.a.
    CostAC = factor['ACPV'] * CPV # A$b p.a.

    Energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
    Loss = np.sum(abs(solution.TDC), axis=0) * DCloss
    Loss = Loss.sum() * pow(10, -9) * resolution / years # PWh p.a.

    LCOE = (CostPV + CostHydro + CostPH + CostDC + CostAC) / (Energy - Loss)
    LCOEPV = CostPV / (Energy - Loss)
    LCOEHydro = CostHydro / (Energy - Loss)
    
    LCOEPH = CostPH / (Energy - Loss)
    LCOEDC = CostDC / (Energy - Loss)
    LCOEAC = CostAC / (Energy - Loss)

    print('Levelised costs of electricity:')
    print('\u2022 LCOE:', LCOE)
    print('\u2022 LCOE-PV:', LCOEPV, '(%s)' % CFPV)
    print('\u2022 LCOE-Hydro & other renewables:', LCOEHydro)
    
    print('\u2022 LCOE-Pumped hydro:', LCOEPH)
    print('\u2022 LCOE-HVDC:', LCOEDC)
    print('\u2022 LCOE-HVAC:', LCOEAC)

    CapDC = solution.CDC * np.array([1350]) * pow(10, -3) # GW-km (1000)
    CapAC = (10 * CPV) * pow(10, -3) # GW-km (1000)

    D = np.zeros((1, 16))

    D[0, :] = [Energy * pow(10, 3), Loss * pow(10, 3),
               CPV, GPV, CapHydro, GHydro, CPHP, CPHS,
               CapDC, CapAC,
               LCOE, LCOEPV, LCOEHydro, LCOEPH, LCOEDC, LCOEAC]

    header = 'Energy,Energy losses,' \
             'Solar capacity [GW],Solar generation [GWh],Existing hydro capacity [GW],Existing hydro generation [GWh],PHES power capacity [GW],PHES energy capacity [GWh],' \
             'DC transmission capacity [GW],AC transmission capacity [GW],' \
             'Levelised cost of electricity (LCOE) [RM/MWh],LCOE Solar [RM/MWh],LCOE Hydro [RM/MWh],LCOE PHES [RM/MWh],LCOE DC [RM/MWh],LCOE AC [RM/MWh]'

    np.savetxt('Results/GGTA{}{}.csv'.format(scenario, percapita), D, fmt='%f', delimiter=',',header=header)
    print('Energy generation, storage and transmission information is produced.')

    return True

def Information(x, flexible):
    """Dispatch: Statistics.Information(x, Hydro)"""

    start = dt.datetime.now()
    print("Statistics start at", start)

    S = Solution(x)
    Deficit = Reliability(S, flexible=flexible)

    try:
        assert Deficit.sum() * resolution - S.allowance < 0.1, 'Energy generation and demand are not balanced.'
    except AssertionError:
        pass
    
    if scenario == "PMY_BMY":
        S.TDC = Transmission(S, output=True) # TDC(t, k), MW
    else:
        S.TDC = np.zeros((intervals, len(DCloss))) # TDC(t, k), MW

        S.MPeak = np.tile(flexible, (nodes, 1)).transpose() # MW
        S.MBaseload = GBaseload.copy() # MW

        S.MPV = S.GPV.sum(axis=1) if S.GPV.shape[1]>0 else np.zeros((intervals, 1))
        
        S.MDischarge = np.tile(S.Discharge, (nodes, 1)).transpose()
        S.MDeficit = np.tile(S.Deficit, (nodes, 1)).transpose()
        S.MCharge = np.tile(S.Charge, (nodes, 1)).transpose()
        S.MStorage = np.tile(S.Storage, (nodes, 1)).transpose()
        S.MSpillage = np.tile(S.Spillage, (nodes, 1)).transpose()

    S.CDC = np.amax(abs(S.TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW
    S.PBMY = [item for sublist in S.TDC for item in sublist]

    S.MHydro = np.tile(S.CHydro - 0.5 * S.EHydro / 8760, (intervals, 1)) * pow(10, 3) # GW to MW
    S.MHydro = np.minimum(S.MHydro, S.MPeak)
    S.MHydro += S.MBaseload # Hydropower & other renewables

    S.MPHS = S.CPHS * np.array(S.CPHP) * pow(10, 3) / sum(S.CPHP)  # GW to MW

    S.Topology = [S.PBMY]

    LPGM(S)
    GGTA(S)

    end = dt.datetime.now()
    print("Statistics took", end - start)

    return True

if __name__ == '__main__':
    capacities = np.genfromtxt('Results/Optimisation_resultxBMY_only5.csv', delimiter=',')
    flexible = np.genfromtxt('Results/Dispatch_FlexibleBMY_only5.csv', delimiter=',', skip_header=1)
    Information(capacities, flexible)