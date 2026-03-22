import sys, os, runpy
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
runpy.run_path(os.path.join(_root, "aula_06", "app_streamlit.py"), run_name="__main__")
