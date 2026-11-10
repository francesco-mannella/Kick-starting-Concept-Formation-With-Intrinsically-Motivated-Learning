import sys, runpy, os, functools
import matplotlib; matplotlib.use("agg")
import matplotlib.animation as an
matplotlib.rcParams["savefig.dpi"] = 200
_pw = an.PillowWriter
class PW(_pw):
    def __init__(self, fps=20, **kw): super().__init__(fps=6, **kw)
an.PillowWriter = PW
sys.argv = ["plot_postures.py"] + sys.argv[1:]
runpy.run_path(os.environ["SRC"] + "/plot_postures.py", run_name="__main__")
