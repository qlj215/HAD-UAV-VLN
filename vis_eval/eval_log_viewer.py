"""Compatibility entrypoint for the Streamlit eval viewer.

The maintained implementation lives in:
    visualize/vis_eval/eval_log_viewer.py

This wrapper keeps the documented command working from the project root:
    streamlit run vis_eval/eval_log_viewer.py -- /path/to/eval_output_dir
"""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parents[1] / "visualize" / "vis_eval" / "eval_log_viewer.py"
runpy.run_path(str(TARGET), run_name="__main__")
