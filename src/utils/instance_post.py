# utils/instance_post.py (새 파일로 두거나 GS3LAM.py 상단에 넣어도 됨)
import torch, numpy as np

@torch.no_grad()
def segment_instances_from_embeddings(sem_logits: torch.Tensor,
                                      inst_embed: torch.Tensor,
                                      alpha: torch.Tensor = None,
                                      sim_thr: float = 0.6,
                                      min_area: int = 30):
    """
    sem_logits: (C,H,W)  - 세만틱 로짓
    inst_embed: (D,H,W)  - 인스턴스 임베딩
    alpha     : (H,W)    - 선택, 투명도 마스크(있으면 신뢰도 필터링)
    return: instance_id_map (H,W) int32, 0=배경/무시
    """
    C, H, W = sem_logits.shape
    D = inst_embed.shape[0]
    sem = sem_logits.argmax(0)  # (H,W)
    emb = inst_embed.permute(1,2,0).reshape(H*W, D)  # (HW,D)
    emb = torch.nn.functional.normalize(emb, dim=1)  # 코사인 거리 안정화
    sem_flat = sem.reshape(-1)

    if alpha is not None:
        valid = (alpha.reshape(-1) > 0.5)
    else:
        valid = torch.ones_like(sem_flat, dtype=torch.bool)

    # 4-이웃 그래프에서 같은 클래스 & 임베딩 유사도>=thr 면 연결
    def idx(y,x): return y*W+x
    parent = torch.arange(H*W, device=sem.device)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    # 이웃 비교
    for y in range(H):
        for x in range(W):
            i = idx(y,x)
            if not valid[i]: continue
            # 오른쪽, 아래만 보면 충분
            if x+1 < W:
                j = idx(y,x+1)
                if valid[j] and sem_flat[i]==sem_flat[j]:
                    sim = (emb[i]*emb[j]).sum()  # 코사인: normalize했으므로 내적
                    if sim >= sim_thr: union(i,j)
            if y+1 < H:
                j = idx(y+1,x)
                if valid[j] and sem_flat[i]==sem_flat[j]:
                    sim = (emb[i]*emb[j]).sum()
                    if sim >= sim_thr: union(i,j)

    # 루트별 id 부여
    root = torch.zeros(H*W, dtype=torch.int64, device=sem.device)
    for i in torch.nonzero(valid).flatten():
        root[i] = find(i)

    # 작은 조각 제거(배경 0으로)
    uniq, counts = torch.unique(root[valid], return_counts=True)
    keep = uniq[counts >= min_area]
    keep_set = set(keep.tolist())

    id_map = torch.zeros(H*W, dtype=torch.int32, device=sem.device)
    next_id = 1
    remap = {}
    for r in uniq.tolist():
        if r in keep_set:
            remap[r] = next_id
            next_id += 1
    mask_keep = torch.tensor([r.item() in remap for r in root], device=sem.device)
    id_map[mask_keep] = torch.tensor([remap.get(int(r),0) for r in root[mask_keep].tolist()], device=sem.device, dtype=torch.int32)

    return id_map.reshape(H,W)
