#!/usr/bin/env python3
"""Leakage-aware multi-asset fine-tuning for Colab T4."""
import os,json,time,random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import stage3_pretrain as s3

SEED=int(os.getenv("ADAPT_SEED","42"))
TRAIN_SYMBOLS=os.getenv("ADAPT_TRAIN_SYMBOLS","EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,EURGBP,GBPJPY,XAUUSD,XAGUSD,BTCUSD,ETHUSD,SP500,NQ,DJ,DAX,FTSE").split(",")
HELD_SYMBOLS=os.getenv("ADAPT_HELD_SYMBOLS","NIKKEI,BRENT,WTI").split(",")
TRAIN_YEARS=range(2015,2022); VAL_YEARS=[2022]; TEST_YEARS=[2023]
OUT=os.getenv("ADAPT_OUT","/content/adaptive_results")
EPOCHS=int(os.getenv("ADAPT_EPOCHS","8")); BATCH=int(os.getenv("ADAPT_BATCH","48"))
STRIDE=int(os.getenv("ADAPT_STRIDE","8")); LR=float(os.getenv("ADAPT_LR","0.00008"))
PATIENCE=int(os.getenv("ADAPT_PATIENCE","3"))
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT,exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
def log(x): print(time.strftime("[%H:%M:%S]"),x,flush=True)

class MultiAssetDS(Dataset):
    def __init__(self,symbols,years,stride):
        self.rows=[]; self.tokens={}; self.labels={}
        for sym in symbols:
            lp=f"{s3.MKT}/lab_{sym}.npz"; tp=f"{s3.MKT}/tok_{sym}_M5.npz"
            if not(os.path.exists(lp) and os.path.exists(tp)): log(f"skip missing {sym}"); continue
            lab=np.load(lp); z=np.load(tp); tok=z["tokens"].astype(np.int64)
            spans=z["spans"].astype(np.int64); bot=np.concatenate(([0],np.cumsum(spans)))[:-1]
            n=min(len(lab["ts"]),int(bot[-1])+1); bot=bot[bot<n]
            tok_of=np.full(n,-1,np.int64); tok_of[bot]=np.arange(len(bot))
            ix=np.maximum.accumulate(np.where(tok_of>=0,np.arange(n),0))
            tok_of=np.where(tok_of>=0,tok_of,ix)
            ts=lab["ts"][:n]; yr=(ts//31557600+1970).astype(np.int32)
            valid=np.isin(yr,years)
            valid &= (lab["nxt"][:n]<2)&(lab["volb"][:n]<4)&(lab["regime"][:n]<6)
            valid &= np.isfinite(lab["fwd12"][:n])&np.isfinite(lab["fwd48"][:n])&np.isfinite(lab["fwd288"][:n])
            valid &= (tok_of>=s3.CTX-6)&(tok_of<len(tok)-1)
            # Purge final 24 hours of each year to prevent longest forward label crossing split.
            for y in years:
                end=int((y+1-1970)*31557600)
                valid &= ~((yr==y)&(ts>=end-86400))
            bars=np.flatnonzero(valid)[::stride]
            self.tokens[sym]=tok; self.labels[sym]=lab
            self.rows.extend((sym,int(b),int(tok_of[b])) for b in bars)
            log(f"{sym} years={sorted(set(yr[bars].tolist()))} windows={len(bars):,}")
        if not self.rows: raise RuntimeError(f"Empty dataset for years={list(years)}")
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        sym,b,en=self.rows[i]; tok=self.tokens[sym]; lab=self.labels[sym]
        st=en-(s3.CTX-6); chunk=tok[st:en+1]
        ids=np.full(s3.CTX,s3.PAD,np.int64)
        ids[:5]=[s3.BOS,s3.SYM0+s3.SYM_ID[sym],s3.TF0+s3.TF_ID["M5"],s3.REG0+int(lab["regime"][b]),s3.SESS0+int(lab["sess"][b])]
        ids[5:5+len(chunk)]=chunk
        q=np.array([lab["fwd12"][b],lab["fwd48"][b],lab["fwd288"][b]],np.float32)
        return torch.from_numpy(ids),int(lab["nxt"][b]),int(lab["volb"][b]),torch.from_numpy(q),sym,int(b)

class HeadModel(nn.Module):
    def __init__(self,base):
        super().__init__(); self.backbone=base; d=s3.N_EMBD
        self.dir_head=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,1))
        self.vol_head=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,4))
        self.q_head=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,9))
    def forward(self,ids):
        x=self.backbone.drop(self.backbone.tok(ids)+self.backbone.pos(torch.arange(ids.shape[1],device=ids.device))[None])
        for block in self.backbone.blocks: x=block(x)
        x=self.backbone.ln_f(x[:,-1])
        return self.dir_head(x).squeeze(-1),self.vol_head(x),self.q_head(x).view(-1,3,3)

def loss_fn(out,yd,yv,yq):
    pd,pv,pq=out; ld=nn.functional.binary_cross_entropy_with_logits(pd,yd.float())
    lv=nn.functional.cross_entropy(pv,yv); qs=torch.tensor([.1,.5,.9],device=pq.device)
    e=yq.unsqueeze(-1)-pq; lq=torch.maximum(qs*e,(qs-1)*e).mean()
    return ld+.5*lv+.5*lq

@torch.no_grad()
def evaluate(model,loader):
    model.eval(); total=0.; n=0; ps=[]; ys=[]
    for ids,yd,yv,yq,_,_ in loader:
        ids=ids.to(DEVICE); yd=yd.to(DEVICE); yv=yv.to(DEVICE); yq=yq.to(DEVICE)
        with torch.autocast("cuda",dtype=torch.float16,enabled=DEVICE.type=="cuda"):
            out=model(ids); loss=loss_fn(out,yd,yv,yq)
        total+=loss.item()*len(ids); n+=len(ids)
        ps.append(torch.sigmoid(out[0]).float().cpu().numpy()); ys.append(yd.cpu().numpy())
    p=np.concatenate(ps); y=np.concatenate(ys)
    try:
        from sklearn.metrics import roc_auc_score
        auc=float(roc_auc_score(y,p))
    except Exception: auc=float("nan")
    return {"loss":total/max(n,1),"acc":float(((p>=.5)==y).mean()),"auc":auc,"brier":float(np.mean((p-y)**2)),"n":int(n)}

def main():
    log(f"device={DEVICE}; train assets={len(TRAIN_SYMBOLS)}; held-out={HELD_SYMBOLS}")
    if DEVICE.type!="cuda": log("WARNING: no CUDA GPU detected")
    tr=MultiAssetDS(TRAIN_SYMBOLS,TRAIN_YEARS,STRIDE)
    va=MultiAssetDS(TRAIN_SYMBOLS,VAL_YEARS,max(STRIDE,16))
    te=MultiAssetDS(TRAIN_SYMBOLS,TEST_YEARS,max(STRIDE,16))
    held=[]
    for sym in HELD_SYMBOLS:
        try: held.append((sym,MultiAssetDS([sym],TEST_YEARS,max(STRIDE,16))))
        except Exception as e: log(f"heldout skip {sym}: {e}")
    tl=DataLoader(tr,batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    vl=DataLoader(va,batch_size=BATCH*2,shuffle=False,num_workers=0,pin_memory=True)
    el=DataLoader(te,batch_size=BATCH*2,shuffle=False,num_workers=0,pin_memory=True)
    ck=os.getenv("CKPT","/content/ckpt/best.pt")
    if not os.path.exists(ck): raise FileNotFoundError(f"Pretrained checkpoint missing: {ck}")
    base=s3.MarketGPT(); base.load_state_dict(torch.load(ck,map_location="cpu",weights_only=True))
    model=HeadModel(base).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.03,betas=(.9,.95))
    scaler=torch.amp.GradScaler("cuda",enabled=DEVICE.type=="cuda")
    best=float("inf"); bad=0; hist=[]
    for ep in range(1,EPOCHS+1):
        model.train(); tot=0.; n=0
        for ids,yd,yv,yq,_,_ in tl:
            ids=ids.to(DEVICE,non_blocking=True); yd=yd.to(DEVICE,non_blocking=True)
            yv=yv.to(DEVICE,non_blocking=True); yq=yq.to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=DEVICE.type=="cuda"):
                out=model(ids); loss=loss_fn(out,yd,yv,yq)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
            tot+=loss.item()*len(ids); n+=len(ids)
        vm=evaluate(model,vl); row={"epoch":ep,"train_loss":tot/max(n,1),"val":vm}; hist.append(row); log(json.dumps(row))
        if vm["loss"]<best: best=vm["loss"]; bad=0; torch.save(model.state_dict(),f"{OUT}/best.pt")
        else:
            bad+=1
            if bad>=PATIENCE: log("early stopping"); break
    model.load_state_dict(torch.load(f"{OUT}/best.pt",map_location=DEVICE,weights_only=True))
    results={"validation":evaluate(model,vl),"test_2023":evaluate(model,el),"heldout_zero_shot":{}}
    for sym,ds in held: results["heldout_zero_shot"][sym]=evaluate(model,DataLoader(ds,batch_size=BATCH*2,shuffle=False))
    results["history"]=hist; results["config"]={"train_symbols":TRAIN_SYMBOLS,"heldout_symbols":HELD_SYMBOLS,"train_years":list(TRAIN_YEARS),"validation_years":VAL_YEARS,"test_years":TEST_YEARS,"stride":STRIDE,"batch":BATCH,"epochs":EPOCHS,"device":str(DEVICE)}
    with open(f"{OUT}/metrics.json","w") as f: json.dump(results,f,indent=2,allow_nan=True)
    log("FINAL "+json.dumps({k:v for k,v in results.items() if k!="history"},allow_nan=True))
    print("ADAPTIVE FINETUNE COMPLETE",flush=True)
if __name__=="__main__": main()
