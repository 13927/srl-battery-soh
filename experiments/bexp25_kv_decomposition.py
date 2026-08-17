"""bexp25: per-unit bias-diversity decomposition, answering whether the two
models are complementary.
MSE_ens = (MSE_t + MSE_g)/2 - E[(pt-pg)^2]/4; the ensemble beats its best member
if and only if diversity/4 > |gap|/2.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery_lab.data_adapters import FIT_UNITS, load_fit_unit
from bexp22_ensemble import duo_predict
OUT = Path("results/battery/bexp25_kv.json")
rows=[]
for uid, ds, bk in FIT_UNITS:
    u=load_fit_unit(ds,bk)
    pt,pg,y=duo_predict(u.train_cells,u.test_cells,seed=0)
    et,eg=np.asarray(pt)-y,np.asarray(pg)-y
    mt,mg=float(np.mean(et**2)),float(np.mean(eg**2))
    div=float(np.mean((np.asarray(pt)-np.asarray(pg))**2))/4
    rows.append({"unit":uid,"rho_resid":float(np.corrcoef(et,eg)[0,1]),
                 "mse_tab":mt,"mse_gbdt":mg,"diversity_over4":div,
                 "gap_over2":abs(mt-mg)/2,"ens_beats_best":bool((mt+mg)/2-div<min(mt,mg))})
    print(rows[-1], flush=True)
json.dump(rows, OUT.open("w"), indent=1)
print("saved", OUT)
