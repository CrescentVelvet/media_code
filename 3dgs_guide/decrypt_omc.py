#!/usr/bin/env python3
"""Decrypt 3DGS_guide OMC files to HiAI IMOD offline models.

The algorithm and byte offsets are recovered from libarenginedemo.so, function
0x1e6d28. CTR decryption is intentionally identical to encryption. This script
accepts an existing extraction of the HAP's resources/rawfile directory.

Requires: python -m pip install cryptography
"""
import argparse
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

KEY_HEX = "6b1d4f8a2c7e9035d84a1f6b3e9c5207a7c2e5904b1d8f3615e9b7d0c4a26f83"
IV_HEX = {
    "s1_part1_of_3_nopose_32.omc": "3f8c1a7d2e5b9046c1d8a3f7b2e6405d",
    "s1_part2_of_3_nopose_32.omc": "7a2d9c4e1b6f8035d4e7a1c9b2f56038",
    "s1_part3_of_3_nopose_32.omc": "b5e1c7a93d2f8460a1c5e9b3d7f2406c",
    "extract.omc": "0c9f4a7e2b5d8136f4a8c2e6b0d39571",
}


def decrypt_file(source, destination, key, initial_counter):
    """Stream AES-256-CTR, preserving file length, and verify HiAI magic."""
    decryptor = Cipher(algorithms.AES(key), modes.CTR(initial_counter)).decryptor()
    digest = hashlib.sha256()
    with source.open("rb") as src, destination.open("wb") as dst:
        first = True
        for block in iter(lambda: src.read(1024 * 1024), b""):
            plain = decryptor.update(block)
            if first:
                if plain[:4] != b"IMOD":
                    raise ValueError(f"{source.name}: invalid plaintext magic {plain[:4]!r}")
                first = False
            digest.update(plain)
            dst.write(plain)
        dst.write(decryptor.finalize())
    return {
        "source": source.name,
        "output": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": digest.hexdigest(),
        "magic": "IMOD",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key = bytes.fromhex(KEY_HEX)
    results = []
    for name, counter_hex in IV_HEX.items():
        source = args.input_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = args.output_dir / (source.stem + ".om")
        result = decrypt_file(source, destination, key, bytes.fromhex(counter_hex))
        results.append(result)
        print(f"PASS {name}: IMOD, {result['bytes']} bytes, sha256={result['sha256']}")
    manifest = args.output_dir / "manifest.json"
    with manifest.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
