#!/usr/bin/env python3
"""Launch this model folder through the shared Ch3Ef Ollama runner."""
from pathlib import Path
import runpy
import sys

model_dir = Path(__file__).resolve().parent
runner = model_dir.parent / "scripts" / "run_ch3ef_full1416_model.py"
sys.argv = [str(runner), "--slug", model_dir.name, *sys.argv[1:]]
runpy.run_path(str(runner), run_name="__main__")
