from torch import nn
import torch.nn.functional as F  # ← 추가

class SemanticDecoder_MLP(nn.Module):
    def __init__(self, feature_out_dim):
        super().__init__()
        self.output_dim = feature_out_dim
        #self.fc1 = nn.Linear(16, 128).cuda()
        #self.fc1 = nn.Linear(128, 128).cuda()
        #self.fc2 = nn.Linear(128, self.output_dim).cuda()

        #self.fc0 = nn.Linear(16, 16).cuda()
        #self.fc1 = nn.Linear(16, 32).cuda()
        #self.fc2 = nn.Linear(32, 64).cuda()
        #self.fc3 = nn.Linear(128, 128).cuda()
        self.fc4 = nn.Linear(128, 256).cuda()

    def forward(self, x):
        input_dim, h, w = x.shape
        x = x.permute(1,2,0).contiguous().view(-1, input_dim) #(16,48,64)->(48,64,16)->(48*64,16)
        #x = torch.relu(self.fc0(x))
        #x = torch.relu(self.fc1(x))
        #x = self.fc2(x)
        #x = torch.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(h, w, self.output_dim).permute(2, 0, 1).contiguous()
        return x

class SemanticDecoder(nn.Module):
    def __init__(self, input_dim, num_classes, emb_dim=32, width=64):
        """
        input_dim: rasterizer가 내보내는 feature 채널 수 (ex. 16)
        num_classes: semantic 클래스 수 (ex. 256)
        emb_dim: instance 임베딩 채널 수 (기본 32)
        width: 백본 중간 채널 수
        """
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(input_dim, width, 3, 1, 1).cuda(), nn.ReLU(True),
            nn.Conv2d(width, width, 3, 1, 1).cuda(), nn.ReLU(True),
        )
        self.sem_head  = nn.Conv2d(width, num_classes, 1).cuda()  # semantic logits
        self.inst_head = nn.Conv2d(width, emb_dim,     1).cuda()  # instance embedding

    def forward(self, x):
        """
        x: (C,H,W) 또는 (1,C,H,W)
        return:
          sem_logits: (num_classes, H, W)
          inst_embed: (emb_dim, H, W)  (L2 normalized)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1,C,H,W)

        feat = self.backbone(x)                  # (1, width, H, W)
        sem_logits = self.sem_head(feat)[0]      # (num_classes, H, W)
        inst_embed = self.inst_head(feat)[0]     # (emb_dim, H, W)
        inst_embed = inst_embed.permute(1, 2, 0)          # (H, W, D)
        inst_embed = F.normalize(inst_embed, dim=-1)       # 픽셀 벡터 L2 정규화
        inst_embed = inst_embed.permute(2, 0, 1).contiguous()  # (D, H, W)

        return sem_logits, inst_embed