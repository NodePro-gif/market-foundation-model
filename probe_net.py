import socket
import urllib.request

for url in [
    "https://datafeed.dukascopy.com/datafeed/XAUUSD/2020/00/02/BID_candles_min_5.bi5",
    "https://www.google.com",
]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as r:
            print("OK", r.status, r.headers.get("Content-Length"), url)
    except Exception as e:
        print("ERR", type(e).__name__, e, url)

print("hostname:", socket.gethostname())
