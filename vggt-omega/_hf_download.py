#!/usr/bin/env python3
"""Download a HuggingFace repo snapshot with SSL verification disabled.

For corporate TLS-intercepting proxies whose CDN endpoints (e.g.
us.aws.cdn.hf.co) present certs that can't be added to the trust store.
Equivalent to `hf download <repo> [file...] --local-dir <dir>` but forces
verify=False.

Usage:
  _hf_download.py <repo_id> <local_dir> [pattern1,pattern2,...]
A 3rd arg (comma-separated allow_patterns) downloads only matching files
(useful for gated repos with multiple large checkpoints).
"""
import os
import sys

# Avoid the Xet/Rust path (it doesn't honor this monkeypatch); use requests.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 1) stdlib https: disable verification.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 2) requests / urllib3: force verify=False on every request; silence warning.
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
_orig = requests.Session.request
def _no_verify(self, *args, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig(self, *args, **kwargs)
requests.Session.request = _no_verify

from huggingface_hub import snapshot_download

if len(sys.argv) < 3:
    sys.exit("usage: _hf_download.py <repo_id> <local_dir> [pattern1,pattern2,...]")

repo_id, local_dir = sys.argv[1], sys.argv[2]
allow_patterns = [p for p in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] else None
kwargs = {"repo_id": repo_id, "local_dir": local_dir}
if allow_patterns:
    kwargs["allow_patterns"] = allow_patterns
print(f"[*] snapshot_download({repo_id!r} -> {local_dir}, allow_patterns={allow_patterns})  [SSL verification DISABLED]")
snapshot_download(**kwargs)
print("[*] done")
