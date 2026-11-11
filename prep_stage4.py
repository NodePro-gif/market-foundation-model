import os, shutil
os.makedirs("/content/ckpt", exist_ok=True)
shutil.copy("/content/ckpt_best.pt", "/content/ckpt/best.pt")
shutil.move("/content/year_XAUUSD_M1.npy", "/content/mkt/year_XAUUSD_M1.npy")
print("ckpt:", os.path.getsize("/content/ckpt/best.pt") / 1e6, "MB")
print("year map:", os.path.exists("/content/mkt/year_XAUUSD_M1.npy"))
print("PREP OK")
