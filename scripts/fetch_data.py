# fetch_data.py — materialize the two TGB edgelists into data/.
# The sweep scripts read plain CSV edgelists; this script downloads each
# dataset once via the py-tgb package and copies the edgelist CSVs into
# data/. Run once before run_trade_sweep.py / run_polecat_sweep.py.
#
#   pip install py-tgb
#   python scripts/fetch_data.py

import os
import shutil
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

WANTED = {
    "tgbn-trade": "tgbn-trade_edgelist.csv",
    "tkgl-polecat": "tkgl-polecat_edgelist.csv",
}


def fetch(name):
    if name.startswith("tgbn"):
        from tgb.nodeproppred.dataset import NodePropPredDataset as DS
    else:
        from tgb.linkproppred.dataset import LinkPropPredDataset as DS
    DS(name=name, root="datasets", preprocess=True)  # triggers download
    hits = glob.glob(os.path.join("datasets", "**", WANTED[name]),
                     recursive=True)
    assert hits, f"{WANTED[name]} not found under datasets/ after download"
    dst = os.path.join(DATA, WANTED[name])
    shutil.copy(hits[0], dst)
    print(f"{name}: copied {hits[0]} -> {dst}")


if __name__ == "__main__":
    for name in WANTED:
        target = os.path.join(DATA, WANTED[name])
        if os.path.exists(target):
            print(f"{name}: already present at {target}")
        else:
            fetch(name)
    print("done")
