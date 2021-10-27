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

    Load, PV, Wind, Inter = (solution.MLoad.sum(axis=1), solution.GPV.sum(axis=1), solution.GWind.sum(axis=1), solution.GInter.sum(axis=1))
    Baseload, Peak = (solution.MBaseload.sum(axis=1), solution.MPeak.sum(axis=1))

    Discharge, Charge, Storage = (solution.Discharge, solution.Charge, solution.Storage)
    Deficit, Spillage = (solution.Deficit, solution.Spillage)

    PHS = solution.CPHS * pow(10, 3) # MWh
    efficiency = solution.efficiency

    for i in range(intervals):
        # Energy supply-demand balance
        assert abs(Load[i] + Charge[i] + Spillage[i]
                   - PV[i] - Wind[i] - Inter[i] - Baseload[i] - Peak[i] - Discharge[i] - Deficit[i]) <= 1

        # Discharge, Charge and Storage
        if i==0:
            assert abs(Storage[i] - 0.5 * PHS + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1
        else:
            assert abs(Storage[i] - Storage[i - 1] + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1

        # Capacity: PV, wind, Discharge, Charge and Storage
        try:
            assert np.amax(PV) <= sum(solution.CPV) * pow(10, 3), print(np.amax(PV) - sum(solution.CPV) * pow(10, 3))
            assert np.amax(Wind) <= sum(solution.CWind) * pow(10, 3), print(np.amax(Wind) - sum(solution.CWind) * pow(10, 3))
            assert np.amax(Inter) <= sum(solution.CInter) * pow(10, 3), print(np.amax(Inter) - sum(solution.CInter) * pow(10, 3))

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
                  solution.MHydro.sum(axis=1), solution.MFossil.sum(axis=1), solution.MInter.sum(axis=1), solution.GPV.sum(axis=1), solution.GWind.sum(axis=1),
                  solution.Discharge, solution.Deficit, -1 * solution.Spillage, -1 * solution.Charge,
                  solution.Storage,
                  solution.AWIJ, solution.ANIT, solution.BNIK, solution.BNPL, solution.BNSG, solution.KHTH,
                  solution.KHVS, solution.CNVH, solution.INMM, solution.IJIK, solution.IJIS, solution.IJIT,
                  solution.IJSG, solution.IKIC, solution.IMIP, solution.IMIC, solution.LATH, solution.LAVH,
                  solution.MYSG, solution.MYTH, solution.MMTH, solution.PLPV, solution.PMPV])
    C = np.around(C.transpose())

    header = 'Operational demand,' \
             'Hydropower & other renewables,Fossil fuels,Import,Solar photovoltaics,Wind,Pumped hydro energy storage,Energy deficit,Energy spillage,PHES-Charge,' \
             'PHES-Storage,' \
             'AWIJ,ANIT,BNIK,BNPL,BNSG,KHTH,KHVS,CNVH,INMM,IJIK,IJIS,IJIT,IJSG,IKIC,IMIP,IMIC,LATH,LAVH,MYSG,MYTH,MMTH,PLPV,PMPV'

    np.savetxt('Results/LPGM_SEAsia{}{}.csv'.format(node, percapita), C, fmt='%f', delimiter=',', header=header, comments='')

    if 'Super' in node:
        header = 'Operational demand,' \
                 'Hydropower & other renewables,Fossil fuels,Import,Solar photovoltaics,Wind,Pumped hydro energy storage,Energy deficit,Energy spillage,' \
                 'Transmission,PHES-Charge,' \
                 'PHES-Storage'

        for j in range(nodes):
            C = np.stack([solution.MLoad[:, j],
                          solution.MHydro[:, j], solution.MFossil[:, j], solution.MInter[:, j], solution.MPV[:, j], solution.MWind[:, j],
                          solution.MDischarge[:, j], solution.MDeficit[:, j], -1 * solution.MSpillage[:, j], solution.Topology[j], -1 * solution.MCharge[:, j],
                          solution.MStorage[:, j]])
            C = np.around(C.transpose())

            np.savetxt('Results/LPGM_{}{}{}.csv'.format(node, percapita, solution.Nodel[j]), C, fmt='%f', delimiter=',', header=header, comments='')

    print('Load profiles and generation mix is produced.')

    return True

def GGTA(solution):
    """GW, GWh, TWh p.a. and A$/MWh information"""

    if node in ['BN', 'SG']:
        factor = np.genfromtxt('Data/factor1.csv', dtype=None, delimiter=',', encoding=None)
    else:
        factor = np.genfromtxt('Data/factor.csv', dtype=None, delimiter=',', encoding=None)
    factor = dict(factor)

    CPV, CWind, CPHP, CPHS, CInter = (sum(solution.CPV), sum(solution.CWind), sum(solution.CPHP), solution.CPHS, sum(solution.CInter)) # GW, GWh
    CapHydro = (CHydro + CGeo + CBio + CWaste).sum() # Hydropower & other resources: GW
    CapFossil = (CCoal + CGas + COil).sum() # Fossil fuels: GW

    GPV, GWind, GHydro, GFossil, GInter = map(lambda x: x * pow(10, -6) * resolution / years,
                                              (solution.GPV.sum(), solution.GWind.sum(), solution.MHydro.sum(), solution.MFossil.sum(), solution.MInter.sum())) # TWh p.a.
    CFPV, CFWind = (GPV / CPV / 8.76, GWind / CWind / 8.76)

    CostPV = factor['PV'] * CPV # A$b p.a.
    CostWind = factor['Wind'] * CWind # A$b p.a.
    CostHydro = factor['Hydro'] * GHydro # A$b p.a.
    CostFossil = factor['Fossil'] * GFossil # A$b p.a.
    CostPH = factor['PHP'] * CPHP + factor['PHS'] * CPHS - factor['LegPH'] # A$b p.a.
    CostInter = factor['Inter'] * CInter # A$b p.a.

    CostDC = np.array([factor['AWIJ'], factor['ANIT'], factor['BNIK'], factor['BNPL'], factor['BNSG'], factor['KHTH'], factor['KHVS'], factor['CNVH'], factor['INMM'], factor['IJIK'], factor['IJIS'], factor['IJIT'], factor['IJSG'], factor['IKIC'], factor['IMIP'], factor['IMIC'], factor['LATH'], factor['LAVH'], factor['MYSG'], factor['MYTH'], factor['MMTH'], factor['PLPV'], factor['PMPV']])
    CostDC = (CostDC * solution.CDC).sum() - factor['LegINTC'] # A$b p.a.
    CostAC = factor['ACPV'] * CPV + factor['ACWind'] * CWind # A$b p.a.

    Energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
    Loss = np.sum(abs(solution.TDC), axis=0) * DCloss
    Loss = Loss.sum() * pow(10, -9) * resolution / years # PWh p.a.

    LCOE = (CostPV + CostWind + CostInter + CostHydro + CostFossil + CostPH + CostDC + CostAC) / (Energy - Loss)
    LCOEPV = CostPV / (Energy - Loss)
    LCOEWind = CostWind / (Energy - Loss)
    LCOEInter = CostInter / (Energy - Loss)
    LCOEHydro = CostHydro / (Energy - Loss)
    LCOEFossil = CostFossil / (Energy - Loss)

    LCOEPH = CostPH / (Energy - Loss)
    LCOEDC = CostDC / (Energy - Loss)
    LCOEAC = CostAC / (Energy - Loss)

    print('Levelised costs of electricity:')
    print('\u2022 LCOE:', LCOE)
    print('\u2022 LCOE-PV:', LCOEPV, '(%s)' % CFPV)
    print('\u2022 LCOE-Wind:', LCOEWind, '(%s)' % CFWind)
    print('\u2022 LCOE-Import:', LCOEInter)
    print('\u2022 LCOE-Hydro & other renewables:', LCOEHydro)
    print('\u2022 LCOE-Fossil fuels:', LCOEFossil)

    print('\u2022 LCOE-Pumped hydro:', LCOEPH)
    print('\u2022 LCOE-HVDC:', LCOEDC)
    print('\u2022 LCOE-HVAC:', LCOEAC)

    CapDC = solution.CDC * np.array([2100, 1000, 900, 1300, 1300, 500, 200, 600, 1000, 900, 1400, 2100, 900, 600, 1000, 1000, 500, 500, 300, 1300, 700, 600, 400]) * pow(10, -3) # GW-km (1000)
    CapDCO = CapDC[[2, 5, 6, 7, 8, 10, 16, 17, 18, 19, 20]].sum() # GW-km (1000)
    CapDCS = CapDC[[0, 1, 3, 4, 9, 11, 12, 13, 14, 15, 21, 22]].sum() # GW-km (1000)
    CapAC = (10 * CPV + 200 * CWind) * pow(10, -3) # GW-km (1000)

    # D = np.zeros((1, 43))
    # D[0, :] = [Energy * pow(10, 3), Loss * pow(10, 3), CPV, GPV, CWind, GWind, CapHydro, GHydro, CInter, GInter, CPHP, CPHS] \
    #           + list(solution.CDC) \
    #           + [LCOE, LCOEPV, LCOEWind, LCOEInter, LCOEHydro, LCOEPH, LCOEDC, LCOEAC]

    D = np.zeros((1, 26))
    D[0, :] = [Energy * pow(10, 3), Loss * pow(10, 3),
               CPV, GPV, CWind, GWind, CapHydro, GHydro, CapFossil, GFossil, CInter, GInter, CPHP, CPHS,
               CapDCO, CapDCS, CapAC,
               LCOE, LCOEPV, LCOEWind, LCOEHydro, LCOEFossil, LCOEInter, LCOEPH, LCOEDC, LCOEAC]

    np.savetxt('Results/GGTA{}{}.csv'.format(node, percapita), D, fmt='%f', delimiter=',')
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

    if 'Super' in node:
        S.TDC = Transmission(S, output=True) # TDC(t, k), MW
    else:
        S.TDC = np.zeros((intervals, len(DCloss))) # TDC(t, k), MW

        S.MPeak = np.tile(flexible, (nodes, 1)).transpose() # MW
        S.MBaseload = GBaseload.copy() # MW

        S.MPV = S.GPV.copy()
        S.MWind = S.GWind.copy() if S.GWind.shape[1]>0 else np.zeros((intervals, 1))
        S.MInter = S.GInter.copy()

        S.MDischarge = np.tile(S.Discharge, (nodes, 1)).transpose()
        S.MDeficit = np.tile(S.Deficit, (nodes, 1)).transpose()
        S.MCharge = np.tile(S.Charge, (nodes, 1)).transpose()
        S.MStorage = np.tile(S.Storage, (nodes, 1)).transpose()
        S.MSpillage = np.tile(S.Spillage, (nodes, 1)).transpose()

    S.CDC = np.amax(abs(S.TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW
    S.AWIJ, S.ANIT, S.BNIK, S.BNPL, S.BNSG, S.KHTH, S.KHVS, S.CNVH, S.INMM, S.IJIK, S.IJIS, S.IJIT, S.IJSG, S.IKIC, S.IMIP, S.IMIC, S.LATH, S.LAVH, S.MYSG, S.MYTH, S.MMTH, S.PLPV, S.PMPV = map(lambda k: S.TDC[:, k], range(S.TDC.shape[1]))

    S.MHydro = np.tile(S.CHydro - 0.5 * S.EHydro / 8760, (intervals, 1)) * pow(10, 3) # GW to MW
    S.MHydro = np.minimum(S.MHydro, S.MPeak)
    S.MFossil = S.MPeak - S.MHydro # Fossil fuels
    S.MHydro += S.MBaseload # Hydropower & other renewables

    S.MPHS = S.CPHS * np.array(S.CPHP) * pow(10, 3) / sum(S.CPHP)  # GW to MW

    # 'AW', 'AN', 'BN', 'KH', 'CN', 'IN', 'IJ', 'IK', 'IM', 'IP', 'IC', 'IS', 'IT', 'LA', 'MY', 'MM', 'PL', 'PM', 'PV', 'SG', 'TH', 'VH', 'VS'
    # S.AWIJ, S.ANIT, S.BNIK, S.BNPL, S.BNSG, S.KHTH, S.KHVS, S.CNVH, S.INMM, S.IJIK, S.IJIS, S.IJIT, S.IJSG, S.IKIC, S.IMIP, S.IMIC, S.LATH, S.LAVH, S.MYSG, S.MYTH, S.MMTH, S.PLPV, S.PMPV
    S.Topology = [-1 * S.AWIJ,
                  -1 * S.ANIT,
                  -1 * S.BNIK - S.BNPL - S.BNSG,
                  -1 * S.KHTH - S.KHVS,
                  -1 * S.CNVH,
                  -1 * S.INMM,
                  S.AWIJ - S.IJIK - S.IJIS - S.IJIT - S.IJSG,
                  S.BNIK + S.IJIK - S.IKIC,
                  -1 * S.IMIP - S.IMIC,
                  S.IMIP,
                  S.IKIC + S.IMIC,
                  S.IJIS,
                  S.ANIT + S.IJIT,
                  -1 * S.LATH - S.LAVH,
                  -1 * S.MYSG - S.MYTH,
                  S.INMM - S.MMTH,
                  S.BNPL - S.PLPV,
                  -1 * S.PMPV,
                  S.PLPV + S.PMPV,
                  S.BNSG + S.IJSG + S.MYSG,
                  S.KHTH + S.LATH + S.MYTH + S.MMTH,
                  S.CNVH + S.LAVH,
                  S.KHVS]

    LPGM(S)
    GGTA(S)

    end = dt.datetime.now()
    print("Statistics took", end - start)

    return True

if __name__ == '__main__':
    capacities = np.genfromtxt('Results/Optimisation_resultxSuper13.csv', delimiter=',')
    flexible = np.genfromtxt('Results/Dispatch_FlexibleSuper13.csv', delimiter=',', skip_header=1)
    Information(capacities, flexible)