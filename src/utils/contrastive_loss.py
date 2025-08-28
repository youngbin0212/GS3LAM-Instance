import torch
import torch.nn.functional as F
import time


class SimpleContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, max_samples=2048):
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples

    def forward(self, embeddings, labels):
        # embeddings: (N, D), labels: (N,)
        N = embeddings.shape[0]
        
        # Too many embeddings → sample only max_samples
        if N > self.max_samples:
            idx = torch.randperm(N)[:self.max_samples]
            embeddings = embeddings[idx]
            labels = labels[idx]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)  # optional but often helpful 

        # (1) 시간 측정 시작
        torch.cuda.synchronize()
        start = time.time()

        # Compute similarity
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # (N_sample, N_sample)

        # Create positive pair mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(embeddings.device)



        # (2) 시간 측정 종료
        torch.cuda.synchronize()
        end = time.time()
        #print(f"[Contrastive Section Time] {end - start:.4f} sec")

        # (3) 디바이스 확인 로그
        #print("embeddings.device:", embeddings.device)
        #print("sim.device:", sim.device)
        #print("labels.device:", labels.device)
        #print("mask.device:", mask.device)

        # Compute loss
        log_prob = F.log_softmax(sim, dim=1)
        loss = -(mask * log_prob).sum(1) / mask.sum(1).clamp(min=1.0)  # avoid division by zero
        return loss.mean()
