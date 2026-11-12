#!/usr/bin/env python3
"""Nested-validation, cost-aware one-bar backtest for multi-asset predictions."""
import os,json
import numpy as np
from sklearn.linear_model import LogisticRegression

MKT=os.getenv("MKT","/content/mkt")
OUT=os.getenv("ADAPT_OUT","/content/adaptive_results")
DEFAULT_SIDE_COST=float(os.getenv("ADAPT_COST_PER_SIDE","0.00015"))
THRESHOLDS=np.arange(.55,.851,.025)
MIN_BETS=int(os.getenv("ADAPT_MIN_BETS","20"))

def load(name):
    z=np.load(f"{OUT}/{name}_predictions.npz",allow_pickle=False)
    return {k:z[k] for k in z.files}

def equity_stats(r):
    if len(r)==0:return {"n":0,"net_total":0.0,"max_drawdown":0.0,"win_rate":None}
    eq=np.cumprod(1+r); peak=np.maximum.accumulate(eq)
    return {"n":int(len(r)),"net_total":float(eq[-1]-1),
        "max_drawdown":float(np.min(eq/peak-1)),"win_rate":float(np.mean(r>0)),
        "mean_bps":float(np.mean(r)*1e4)}

def main():
    va,te=load("validation"),load("test")
    costs=json.loads(os.getenv("ADAPT_COST_MAP","{}"))
    symbols=sorted(set(va["symbols"].tolist()) & set(te["symbols"].tolist()))
    report={"cost_assumption":"fractional return per side; override with ADAPT_COST_MAP",
            "per_symbol":{},"thresholds":THRESHOLDS.tolist()}
    for sym in symbols:
        vm=va["symbols"]==sym; tm=te["symbols"]==sym
        vp,vy,vb=va["p"][vm],va["y"][vm],va["bars"][vm]
        tp,ty,tb=te["p"][tm],te["y"][tm],te["bars"][tm]
        order=np.argsort(vb); vp,vy,vb=vp[order],vy[order],vb[order]
        cut=int(len(vp)*.60)
        if cut<50 or len(vp)-cut<MIN_BETS or len(tp)==0:
            report["per_symbol"][sym]={"status":"INSUFFICIENT_VALIDATION_OR_TEST","n_val":len(vp),"n_test":len(tp)}
            continue
        # Fit probability calibration only on the first chronological validation segment.
        cal=LogisticRegression(C=1.0,max_iter=1000).fit(
            np.log(np.clip(vp[:cut],1e-5,1-1e-5)/(1-np.clip(vp[:cut],1e-5,1-1e-5))).reshape(-1,1),vy[:cut])
        def calibrated(p):
            x=np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5))).reshape(-1,1)
            return cal.predict_proba(x)[:,1]
        val_p=calibrated(vp[cut:]); test_p=calibrated(tp)
        lab=np.load(f"{MKT}/lab_{sym}.npz")
        close=lab["c"].astype(np.float64)
        def fwd(b):
            b=np.asarray(b,dtype=np.int64); ok=(b>=0)&(b+1<len(close))
            r=np.full(len(b),np.nan); r[ok]=(close[b[ok]+1]-close[b[ok]])/close[b[ok]]
            return r
        val_r=fwd(vb[cut:]); test_r=fwd(tb)
        side=float(costs.get(sym,DEFAULT_SIDE_COST)); roundtrip=2*side
        best=None
        for threshold in THRESHOLDS:
            sig=np.where(val_p>=threshold,1,np.where(val_p<=1-threshold,-1,0))
            mask=(sig!=0)&np.isfinite(val_r)
            if mask.sum()<MIN_BETS:continue
            net=sig[mask]*val_r[mask]-roundtrip
            score=float(np.mean(net))
            if best is None or score>best["score"]:
                best={"threshold":float(threshold),"score":score,"validation":equity_stats(net)}
        if best is None:
            report["per_symbol"][sym]={"status":"NO_VALIDATED_EDGE","n_val":len(vp),"n_test":len(tp)}
            continue
        sig=np.where(test_p>=best["threshold"],1,np.where(test_p<=1-best["threshold"],-1,0))
        mask=(sig!=0)&np.isfinite(test_r)
        net=sig[mask]*test_r[mask]-roundtrip
        report["per_symbol"][sym]={"status":"TESTED","threshold_locked":best["threshold"],
            "validation_threshold_selection":best["validation"],"test_2023":equity_stats(net),
            "cost_per_side":side,"round_trip_cost":roundtrip,
            "test_coverage":float(mask.mean()) if len(mask) else 0.0}
    with open(f"{OUT}/backtest_cost_aware.json","w") as f:json.dump(report,f,indent=2,allow_nan=True)
    print(json.dumps(report,indent=2,allow_nan=True))
if __name__=="__main__":main()
