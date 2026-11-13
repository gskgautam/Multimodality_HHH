#!/usr/bin/env python3
from pathlib import Path
import runpy, sys
model_dir = Path(__file__).resolve().parent
runner = model_dir / "run_model.py"
sys.argv = [str(runner), "--slug", model_dir.name, *sys.argv[1:]]
runpy.run_path(str(runner), run_name="__main__")
