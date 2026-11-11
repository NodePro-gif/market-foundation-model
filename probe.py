import torch
print("cuda:", torch.cuda.is_available(), flush=True)
try:
    import __main__
    print("still training?" , flush=True)
except Exception:
    pass
print("PROBE OK", flush=True)
