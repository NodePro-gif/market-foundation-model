import shutil, os
shutil.move("/content/year_XAUUSD_M1.npy", "/content/mkt/year_XAUUSD_M1.npy")
print("moved:", os.path.getsize("/content/mkt/year_XAUUSD_M1.npy"))
