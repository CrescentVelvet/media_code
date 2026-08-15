#!/usr/bin/env python3
"""Extract the corporate proxy's TLS cert chain and build a CA bundle.

Behind a TLS-intercepting corporate proxy, pip/hf/git fail SSL because the
proxy's CA isn't trusted. This script:
  1. CONNECTs to a target (default huggingface.co) through the proxy, using
     auth from http_proxy/https_proxy.
  2. Captures the cert chain the proxy presents.
  3. Builds ~/.ca-bundle.crt = system bundle + extra CA files + captured chain.
  4. Self-tests whether the bundle now verifies the proxy's cert.

Run via setup_ca_bundle.sh (which sources _env.sh for proxy + conda).
"""
import base64
import os
import socket
import ssl
import sys
import urllib.parse
import urllib.request

TARGET = os.environ.get("CA_PROBE_HOST", "huggingface.co")
PORT = 443
OUT = os.path.expanduser("~/.ca-bundle.crt")
SYS_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"
EXTRA_CA_DIRS = [
    "/usr/local/share/ca-certificates",            # Debian/Ubuntu
    "/etc/pki/ca-trust/source/anchors",            # RHEL/CentOS/Fedora
    "/etc/ca-certificates/trust-source/anchors",   # Arch
]


def pick_proxy():
    for k in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        v = os.environ.get(k)
        if v:
            return v
    return None


def get_chain(phost, pport, auth):
    s = socket.create_connection((phost, pport), timeout=30)
    req = f"CONNECT {TARGET}:{PORT} HTTP/1.1\r\nHost: {TARGET}:{PORT}\r\n"
    if auth:
        req += f"Proxy-Authorization: {auth}\r\n"
    req += "\r\n"
    s.sendall(req.encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            sys.exit("ERROR: proxy closed the connection during CONNECT")
        buf += chunk
    status = buf.split(b"\r\n")[0].decode(errors="replace")
    if " 200 " not in status:
        sys.exit(f"ERROR: CONNECT failed: {status.strip()}")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        ss = ctx.wrap_socket(s, server_hostname=TARGET)
    except Exception as e:
        sys.exit(f"ERROR: TLS handshake with {TARGET} via proxy failed: {e}")

    chain = []
    if hasattr(ss, "get_unverified_chain"):
        try:
            chain = ss.get_unverified_chain() or []
        except Exception:
            chain = []
    if not chain:
        leaf = ss.getpeercert(binary_form=True)
        if leaf:
            chain = [leaf]
    return chain


def read_pem(path):
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def main():
    proxy = pick_proxy()
    if not proxy:
        sys.exit("ERROR: no http_proxy/https_proxy in env. Fill proxy.env / source _env.sh first.")
    if "://" not in proxy:
        proxy = "http://" + proxy
    u = urllib.parse.urlparse(proxy)
    phost = u.hostname
    pport = u.port or 8080
    auth = ""
    if u.username and u.password:
        pair = f"{urllib.parse.unquote(u.username)}:{urllib.parse.unquote(u.password)}"
        auth = "Basic " + base64.b64encode(pair.encode()).decode()

    print(f"[*] probing https://{TARGET} via proxy {phost}:{pport} ...")
    chain = get_chain(phost, pport, auth)
    if not chain:
        sys.exit("ERROR: no certificate retrieved from the proxy")
    captured = "".join(ssl.DER_cert_to_PEM_cert(c) for c in chain)
    print(f"[*] captured {len(chain)} cert(s) from the handshake")

    parts = []
    if os.path.isfile(SYS_BUNDLE):
        parts.append(read_pem(SYS_BUNDLE))
        print(f"[*] included system bundle: {SYS_BUNDLE}")
    n_extra = 0
    for d in EXTRA_CA_DIRS:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith((".crt", ".pem")):
                    parts.append(read_pem(os.path.join(d, f)))
                    n_extra += 1
    if n_extra:
        print(f"[*] included {n_extra} extra CA file(s) from system drop-in dirs")
    parts.append(captured)

    with open(OUT, "w") as f:
        f.write("".join(parts))
    print(f"[*] wrote CA bundle: {OUT} ({os.path.getsize(OUT)} bytes)")

    os.environ["REQUESTS_CA_BUNDLE"] = OUT
    os.environ["SSL_CERT_FILE"] = OUT
    ok, err = False, ""
    try:
        import requests
        requests.get(f"https://{TARGET}/", timeout=30)
        ok = True
    except ImportError:
        try:
            urllib.request.urlopen(f"https://{TARGET}/", timeout=30)
            ok = True
        except Exception as e:
            err = str(e)
    except Exception as e:
        err = str(e)

    if ok:
        print(f"[OK] bundle verifies https://{TARGET} — SSL should work now.")
        print(f"     _env.sh will auto-use {OUT}; rerun 01_download_models.sh.")
    else:
        print(f"[FAIL] bundle does NOT verify https://{TARGET}: {err}")
        print("       The proxy likely didn't send its root CA in the handshake.")
        print("       Obtain the corporate root CA (.crt) and append it to:")
        print(f"         {OUT}")
        print("       then rerun 01_download_models.sh.")


if __name__ == "__main__":
    main()
