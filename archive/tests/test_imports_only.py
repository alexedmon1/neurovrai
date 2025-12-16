#!/usr/bin/env python3
"""Test imports only - no execution"""
import sys
print("1. Starting import test", flush=True)
sys.stdout.flush()

print("2. Importing Path...", flush=True)
from pathlib import Path
print("   ✓ Path imported", flush=True)

print("3. Importing load_config...", flush=True)
from neurovrai.config import load_config
print("   ✓ load_config imported", flush=True)

print("4. Importing run_func_preprocessing...", flush=True)
from neurovrai.preprocess.workflows.func_preprocess import run_func_preprocessing
print("   ✓ run_func_preprocessing imported", flush=True)

print("\n✓ All imports successful!", flush=True)
