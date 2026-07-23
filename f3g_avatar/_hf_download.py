#!/usr/bin/env python3
"""Download a file from a HuggingFace repo with SSL verification disabled.

For corporate TLS-intercepting proxies whose CDN endpoints present certs that
can't be added to the trust store. Equivalent to `hf download <repo> <file>
--local-dir <dir>` but forces verify=False.

Args:
  repo_id        HF repo id, e.g. "wjmenu/F3G-avatar"
  filename       file path within the repo, e.g. "checkpoints/avatarrex_zzr/epoch_latest.pt"
  local_dir      destination directory
  token (opt)    HF token (for gated repos)
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

from huggingface_hub import hf_hub_download

if len(sys.argv) < 4:
    sys.exit("usage: _hf_download.py <repo_id> <filename> <local_dir> [token]")

repo_id, filename, local_dir = sys.argv[1], sys.argv[2], sys.argv[3]
token = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4].strip() else None
extra = {}
if token:
    extra["token"] = token
print(f"[*] hf_hub_download({repo_id!r}, {filename!r} -> {local_dir})  [SSL verification DISABLED]"
      + ("  token=***" if token else ""))
hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, **extra)
print("[*] done")
