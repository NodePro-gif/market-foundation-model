import os, json
ck = "/content/ckpt"
if os.path.isdir(ck):
    for f in sorted(os.listdir(ck)):
        print(f, os.path.getsize(os.path.join(ck, f)) / 1e6, "MB")
else:
    print("no ckpt dir")
if os.path.exists("/content/pretrain_log.json"):
    print(json.load(open("/content/pretrain_log.json")))
print("PROBE OK")
