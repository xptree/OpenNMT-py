#!/usr/bin/env python
# encoding: utf-8
# File Name: build_molecule_storage.py
# Author: Jiezhong Qiu
# Create Time: 2020/04/09 12:27
# TODO:

import json
import dgl
import dgl.data
from tqdm import tqdm
import logging
import argparse
from joblib import Parallel, delayed
from onmt.inputters.chem_util import openbabel_to_dgl_graph

fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def wrapped_to_dgl(smi):
    g = openbabel_to_dgl_graph(smi)
    g.readonly()
    return smi, g

def build_molecule_storage(src_files, save, num_workers):
    src = []
    for src_file in src_files:
        with open(src_file, "r") as f:
            content = f.readlines()
        src += content
    logger.info("%d molecules" % len(src))
    src = ["".join(x.strip().split(" ")[1:]) for x in src]
    src = list(set(src))
    logger.info("%d unique molecules" % len(src))
    src = Parallel(n_jobs=num_workers, backend='loky')(delayed(wrapped_to_dgl)(x) for x in tqdm(src, ascii=True))
    smi_list, g_list = list(zip(*src))
    dgl.data.utils.save_graphs(save + ".bin", g_list, labels=None)
    with open(save + ".json", "w") as f:
        json.dump(smi_list, f, indent=4)

if __name__ == "__main__":
    handler = logging.FileHandler("preprocess.log", 'w')
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    parser = argparse.ArgumentParser("argument for preprocessing moleculear data")
    parser.add_argument('--src-files','--list', nargs='+', help='src files', required=True)
    parser.add_argument("--num-workers", type=int, default=32, help="multiprocessing")
    parser.add_argument("--save", type=str, required=True, help="filename to save processed dgl graphs")

    args = parser.parse_args()
    build_molecule_storage(args.src_files, args.save, args.num_workers)
