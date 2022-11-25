# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from Optimisation import scenario

def Transmission(solution, output=False):
    """TDC = Network.Transmission(S)"""

    Nodel, PVl = (solution.Nodel, solution.PVl)
    intervals, nodes = (solution.intervals, solution.nodes)

    MPV = list(map(np.zeros, [(nodes, intervals)]))[0]

    for i, j in enumerate(Nodel):
        try:
            MPV[i, :] = solution.GPV[:, np.where(PVl==j)[0]].sum(axis=1)
        except:
            continue
    MPV = MPV.transpose() # Sij-GPV(t, i), MW


    MBaseload = solution.GBaseload # MW
    CPeak = solution.CPeak # GW
    pkfactor = np.tile(CPeak, (intervals, 1)) / CPeak.sum()
    MPeak = np.tile(solution.flexible, (nodes, 1)).transpose() * pkfactor # MW

    MLoad = solution.MLoad # EOLoad(t, j), MW

    defactor = MLoad / MLoad.sum(axis=1)[:, None]
    MDeficit = np.tile(solution.Deficit, (nodes, 1)).transpose() * defactor # MDeficit: EDE(j, t)

    spfactor = np.divide(MPV, MPV.sum(axis=1)[:, None], where=MPV.sum(axis=1)[:, None]!=0)
    MSpillage = np.tile(solution.Spillage, (nodes, 1)).transpose() * spfactor # MSpillage: ESP(j, t)

    CPHP = solution.CPHP
    pcfactor = np.tile(CPHP, (intervals, 1)) / sum(CPHP) if sum(CPHP) != 0 else 0
    MDischarge = np.tile(solution.Discharge, (nodes, 1)).transpose() * pcfactor # MDischarge: DPH(j, t)
    MCharge = np.tile(solution.Charge, (nodes, 1)).transpose() * pcfactor # MCharge: CHPH(j, t)

    MImport = MLoad + MCharge + MSpillage - MPV - MBaseload - MPeak - MDischarge - MDeficit  # EIM(t, j), MW

    PBMY = -1 * MImport[:, np.where(Nodel == 'PMY')[0][0]] if scenario == "PMY_BMY" else np.zeros(intervals) # Exports from peninsular Malaysia to Borneo

    TDC = np.array([PBMY]).transpose() # TDC(t, k), MW

    if output:
        MStorage = np.tile(solution.Storage, (nodes, 1)).transpose() * pcfactor # SPH(t, j), MWh
        solution.MPV, solution.MBaseload, solution.MPeak = (MPV, MBaseload, MPeak)
        solution.MDischarge, solution.MCharge, solution.MStorage = (MDischarge, MCharge, MStorage)
        solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage)

    return TDC