import glob
import os
from typing import Optional

import numpy as np
import torch
from natsort import natsorted

from src.datasets.basedataset import GradSLAMDataset

class ScannetSemanticDataset(GradSLAMDataset):
    # ⛳️ 실험할 때 여기 2줄만 수동으로 바꾸고 run.py 두 번 실행
    # (A) ScanNet: semantic="label-filt", instance="instance"
    # (B) YOLO-SAM: semantic="gt_sem_yolo11sam_png", instance="gt_inst_yolo11sam_png"
    #SEMANTIC_DIR = "label-filt"
    #INSTANCE_DIR = "instance"
    #SEMANTIC_DIR = "gt_sem_yolo11sam_png"
    #INSTANCE_DIR = "gt_inst_yolo11sam_png"
    #SEMANTIC_DIR = "ysd_gt_sem_png"
    #INSTANCE_DIR = "ysd_gt_inst_png"
    SEMANTIC_DIR = "pan_gt_sem_png"
    INSTANCE_DIR = "pan_gt_inst_png"

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 968,
        desired_width: Optional[int] = 1296,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
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

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        #object_paths = natsorted(glob.glob(f"{self.input_folder}/label-filt/*.png"))

        def stem(p): return os.path.splitext(os.path.basename(p))[0]
        def build(dir_name, ext): 
            return [os.path.join(self.input_folder, dir_name, f"{stem(c)}.{ext}") for c in color_paths]

        # semantic(object) — 반드시 있어야 함
        object_paths = build(self.SEMANTIC_DIR, "png")
        if not all(os.path.exists(p) for p in object_paths):
            missing = [stem(c) for c,p in zip(color_paths, object_paths) if not os.path.exists(p)]
            raise FileNotFoundError(f"[semantic] '{self.SEMANTIC_DIR}' 누락 {len(missing)}개, 예) {missing[:5]}")

        # instance — 반드시 있어야 함(이번 연구는 instance가 핵심이므로)
        instance_paths = build(self.INSTANCE_DIR, "png")
        if not all(os.path.exists(p) for p in instance_paths):
            missing = [stem(c) for c,p in zip(color_paths, instance_paths) if not os.path.exists(p)]
            raise FileNotFoundError(f"[instance] '{self.INSTANCE_DIR}' 누락 {len(missing)}개, 예) {missing[:5]}")

        # 개수/파일명 정합 체크
        assert len(color_paths) == len(depth_paths) == len(object_paths) == len(instance_paths) and len(color_paths) > 0
        assert all(stem(c)==stem(d)==stem(o)==stem(i)
                   for c,d,o,i in zip(color_paths, depth_paths, object_paths, instance_paths)), \
               "Filename mismatch among color/depth/object/instance."

        # ------------------------------  
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
  
        #return color_paths, depth_paths, object_paths, embedding_paths
        # 반환 튜플 길이를 5개로 고정 (instance_paths 추가)
        return color_paths, depth_paths, object_paths, instance_paths, embedding_paths
    

    def load_poses(self):
        poses = []
        posefiles = natsorted(glob.glob(f"{self.input_folder}/pose/*.txt"))
        for posefile in posefiles:
            _pose = torch.from_numpy(np.loadtxt(posefile))
            poses.append(_pose)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        print(embedding_file_path)
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
    