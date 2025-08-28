# src/datasets/replica.py

import glob
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from natsort import natsorted

from src.datasets.basedataset import GradSLAMDataset


class ReplicaSemanticDataset(GradSLAMDataset):
    """
    Replica dataset loader (semantic + instance).
    Expects a layout like:

        basedir/
          └── office0/
               ├── rgb/ or color/ or results/
               ├── depth/ or results/
               ├── semantic_class/              # semantic label PNGs
               └── inst/inst_png/               # instance label PNGs

    Both semantic and instance folders may contain files named like:
        semantic_class_0.png, semantic_class_1.png, ...

    This class resolves label paths by:
      1) trying one-to-one basename mapping: <rgb_stem>.png
      2) falling back to index-based mapping (preferred): semantic_class_{idx}.png
         where idx is the trailing integer parsed from the RGB stem
    """

    def __init__(
        self,
        config_dict,
        basedir: str,
        sequence: str,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        print("Load Replica dataset!!!")
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")

        # Semantic dir (original vMAP semantic labels)
        self.sem_dir = os.path.join(self.input_folder, "semantic_class")

        # Instance dir (your generated instance masks)
        # Primary location you reported:
        primary_inst = os.path.join(self.input_folder, "inst", "inst_png")
        # Some codebases used alternative layouts; we keep them as fallbacks.
        #alt_inst_a = os.path.join(self.input_folder, "gt_replica", "inst_png")
        #alt_inst_b = os.path.join(self.input_folder, "gt_inst", "inst_png")

        inst_candidates = [primary_inst] #, alt_inst_a, alt_inst_b]
        self.ins_dir = next((d for d in inst_candidates if os.path.isdir(d)), primary_inst)

        if not os.path.isdir(self.sem_dir):
            print(f"[warn] semantic_class dir not found at: {self.sem_dir}")
        if not os.path.isdir(self.ins_dir):
            tried = "\n  ".join(inst_candidates)
            print(f"[warn] instance dir not found. tried:\n  {tried}")
        else:
            print(f"[ok] using instance dir: {self.ins_dir}")

        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    # ---------- Helpers ----------

    def _glob_sorted(self, pattern: str) -> List[str]:
        return natsorted(glob.glob(pattern))

    def _rgb_paths(self) -> List[str]:
        # try common RGB locations in priority order
        return (
            self._glob_sorted(f"{self.input_folder}/rgb/*.jpg")
            or self._glob_sorted(f"{self.input_folder}/rgb/*.png")
            or self._glob_sorted(f"{self.input_folder}/color/*.jpg")
            or self._glob_sorted(f"{self.input_folder}/color/*.png")
            or self._glob_sorted(f"{self.input_folder}/results/frame*.jpg")
            or self._glob_sorted(f"{self.input_folder}/results/frame*.png")
        )

    def _depth_paths(self) -> List[str]:
        # try common depth locations in priority order
        return (
            self._glob_sorted(f"{self.input_folder}/depth/*.png")
            or self._glob_sorted(f"{self.input_folder}/depth/*.jpg")
            or self._glob_sorted(f"{self.input_folder}/results/depth*.png")
            or self._glob_sorted(f"{self.input_folder}/results/depth*.jpg")
        )

    @staticmethod
    def _stem(p: str) -> str:
        return os.path.splitext(os.path.basename(p))[0]

    def _abspath_in_input(self, dir_path: str) -> str:
        return dir_path if os.path.isabs(dir_path) else os.path.join(self.input_folder, dir_path)

    def _build_same_stem(self, dir_path: str, ext: str, color_paths: List[str]) -> List[str]:
        """One-to-one mapping: <rgb_stem>.ext"""
        base = self._abspath_in_input(dir_path)
        return [os.path.join(base, f"{self._stem(c)}.{ext}") for c in color_paths]

    def _build_by_index(
        self,
        dir_path: str,
        ext: str,
        color_paths: List[str],
        index_offset: int = 0,
        pad: int = 6,
    ) -> List[str]:
        """
        Fallback resolver:
        Extract trailing integer from RGB stem and try variants:
          semantic_class_{idx}.ext  (preferred)
          {idx}.ext
          semantic_class_{idx:0{pad}d}.ext
          {idx:0{pad}d}.ext
        then finally <rgb_stem>.ext as last resort.
        """
        base = self._abspath_in_input(dir_path)
        out = []
        for c in color_paths:
            st = Path(c).stem
            m = re.search(r"(\d+)$", st)
            idx = int(m.group(1)) + index_offset if m else None

            candidates = []
            if idx is not None:
                candidates.append(os.path.join(base, f"semantic_class_{idx}.{ext}"))                 # preferred (no pad)
                candidates.append(os.path.join(base, f"{idx}.{ext}"))                                 # plain index
                candidates.append(os.path.join(base, f"semantic_class_{idx:0{pad}d}.{ext}"))         # padded
                candidates.append(os.path.join(base, f"{idx:0{pad}d}.{ext}"))                         # padded plain

            # last resort: original rgb stem
            candidates.append(os.path.join(base, f"{st}.{ext}"))

            chosen = next((p for p in candidates if os.path.exists(p)), None)
            if chosen is None:
                # helpful debug if anything is missing
                print(
                    f"[warn] no match for color '{Path(c).name}' in '{base}', "
                    f"tried: {', '.join(Path(x).name for x in candidates)}"
                )
                chosen = candidates[0]
            out.append(chosen)
        return out

    # ---------- Required API ----------

    def get_filepaths(self) -> Tuple[List[str], List[str], List[str], List[str], Optional[List[str]]]:
        color_paths = self._rgb_paths()
        if not color_paths:
            raise FileNotFoundError("[Replica] RGB 프레임을 찾지 못했습니다. rgb/ 또는 color/ 또는 results/ 확인")

        depth_paths = self._depth_paths()
        if not depth_paths:
            raise FileNotFoundError("[Replica] Depth 프레임을 찾지 못했습니다. depth/ 또는 results/ 확인")

        # First try simple same-stem mapping (<rgb_stem>.png)
        object_paths = self._build_same_stem(self.sem_dir, "png", color_paths)
        instance_paths = self._build_same_stem(self.ins_dir, "png", color_paths)

        # If any are missing, fall back to index-based resolution (semantic_class_{idx}.png, etc.)
        missing_sem = [p for p in object_paths if not os.path.exists(p)]
        missing_ins = [p for p in instance_paths if not os.path.exists(p)]
        if missing_sem or missing_ins:
            obj2 = self._build_by_index(self.sem_dir, "png", color_paths)
            ins2 = self._build_by_index(self.ins_dir, "png", color_paths)

            # adopt fallback only if it fully resolves
            if all(os.path.exists(p) for p in obj2):
                object_paths = obj2
            else:
                bad_o = [p for p in obj2 if not os.path.exists(p)][:5]
                if bad_o:
                    print("[err] missing semantic samples (first 5):", bad_o)
            if all(os.path.exists(p) for p in ins2):
                instance_paths = ins2
            else:
                bad_i = [p for p in ins2 if not os.path.exists(p)][:5]
                if bad_i:
                    print("[err] missing instance samples (first 5):", bad_i)

        # Sanity: lengths must match
        assert len(color_paths) == len(depth_paths) == len(object_paths) == len(instance_paths) and len(color_paths) > 0, \
            f"len mismatch: color={len(color_paths)}, depth={len(depth_paths)}, sem={len(object_paths)}, inst={len(instance_paths)}"

        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))

        print(
            f"[replica] counts — color:{len(color_paths)} depth:{len(depth_paths)} "
            f"sem:{len(object_paths)} inst:{len(instance_paths)}"
        )
        if instance_paths:
            print("[replica] instance sample:", instance_paths[0])

        return color_paths, depth_paths, object_paths, instance_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        # Keep embeddings CPU-safe for portability
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
