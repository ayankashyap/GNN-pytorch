import os
import os.path as osp
import urllib
import zipfile
import ssl
import pickle

import logging
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ZINC(Dataset):

    url = "https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1"
    split_url = (
        "https://raw.githubusercontent.com/graphdeeplearning/"
        "benchmarking-gnns/master/data/molecules/{}.index"
    )

    def __init__(self, root_dir: str, split: str = "train", edge_attrs: bool = False):
        assert split in ["train", "val", "test"]
        self.root_dir = root_dir
        self.edge_attrs = edge_attrs
        self.mol_dir = osp.join(self.root_dir, "molecules")
        self.processed_dir = osp.join(self.root_dir, "processed_mol")
        path = osp.join(self.processed_dir, f"{split}.pt")
        if not osp.isfile(path):
            self.download()
            self.process()
        self.data = torch.load(path)

    def __getitem__(self, idx):
        x,adj,y = self.data[idx]
        if not self.edge_attrs:
            # If we are not using edge embeddings, then convert
            # the adjacency matrix tensor to float and remove the 
            # bond types to only show connections
            adj = adj.to(torch.float)
            adj = adj.masked_fill(adj>0, 1) 
        return x,adj,y


    def __len__(self):
        return len(self.data)

    def download(self):
        path = download_url(self.url, self.root_dir)
        logger.info(f'Unzipping {path}')
        with zipfile.ZipFile(path, "r") as f:
            f.extractall(self.root_dir)
        os.unlink(path)
        # Dont think we need to download indices

    def process(self):
        os.makedirs(self.processed_dir)
        for split in ["train", "val", "test"]:
            logger.info(f'Unpickling {split}.pickle')
            with open(osp.join(self.mol_dir, f"{split}.pickle"), "rb") as f:
                mols = pickle.load(f)

            indices = range(len(mols))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f"Processing {split} dataset")

            data_list = []
            for idx in indices:
                mol = mols[idx]

                x = mol["atom_type"].to(torch.long).view(1, -1)
                y = mol["logP_SA_cycle_normalized"].to(torch.float).view(1,1)
                # dont change this to torch float because we might need embeddings for bond types
                adj = mol["bond_type"] 

                data_list.append((x, adj, y))
                pbar.update(1)

            pbar.close()
            sp_pth = osp.join(self.processed_dir, f"{split}.pt")
            torch.save(data_list, sp_pth)
            logger.info(f"Saving to {sp_pth}")


def download_url(url: str, folder: str, filename: str=None):

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if osp.exists(path):
        logger.info(f"Using existing {filename}")
        return path

    logger.info(f"Downloading {url}")

    if not osp.isdir(folder):
        os.makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path
