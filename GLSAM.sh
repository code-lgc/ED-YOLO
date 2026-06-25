class EfficientSparseAttention(nn.Module):
    """
    GLSA: Global-Local Sparse Attention
    1. Local分支: 使用 DWConv 捕捉 Car/Person 的边缘细节。
    2. Global分支: 使用下采样 K/V 的稀疏注意力，捕捉 Bus/Truck 和路面的长距离关联。
    """
    def __init__(self, dim, num_heads=8, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # --- 关键改进 1: 局部捕获分支 (针对 Car/Person) ---
        # 使用 5x5 DWConv 替代普通的 3x3，增大感受野同时保持局部性
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)

        # --- 关键改进 2: 线性投影 ---
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        # --- 关键改进 3: 稀疏化 Key/Value (Soft Sparsity) ---
        # 不做 TopK 截断，而是通过空间压缩实现“计算稀疏”
        # 这样不会丢失小目标的梯度，sr_ratio 控制稀疏程度
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 融合系数 (可学习)
        self.local_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.global_weight = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # === Branch A: Local Context (保护 Car 的精度) ===
        # 这一步保证了即使 Attention 学得不好，基础的卷积特征还在，防止掉点
        x_local = self.local_conv(x)

        # === Branch B: Global Sparse Attention ===
        # 1. 准备 Query
        q = self.q(x.flatten(2).transpose(1, 2)) # (B, N, C)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 2. 准备 Key, Value (稀疏化)
        if self.sr_ratio > 1:
            # 空间下采样 -> 稀疏代表点
            x_ = self.sr(x).reshape(B, C, -1).transpose(1, 2) # (B, N_sparse, C)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x.flatten(2).transpose(1, 2)).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            
        k, v = kv[0], kv[1] # (B, Heads, N_sparse, Head_Dim)

        # 3. Attention 计算 (Global Context)
        # 这里的计算量是 N * N_sparse，远小于 N * N
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_global = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_global = self.proj(x_global)
        x_global = self.proj_drop(x_global)
        x_global = x_global.transpose(1, 2).reshape(B, C, H, W)

        # === Fusion: 自适应融合 ===
        # 不要简单的相加，给予网络动态调整权重的能力
        return (x_local * self.local_weight) + (x_global * self.global_weight)


class Bottleneck_GLSA(nn.Module):
    """
    集成 GLSA 的 Bottleneck
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        
        # 自动调整 num_heads
        num_heads = 8
        while c_ % num_heads != 0 and num_heads > 1:
            num_heads //= 2
            
        # sr_ratio 可以根据层级调整，深层(P5)可以用1，浅层(P3/P4)建议用2或4
        self.attn = EfficientSparseAttention(c_, num_heads=num_heads, sr_ratio=2)
        
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        y = self.attn(y)
        y = self.cv2(y)
        return x + y if self.add else y


class C3k2_GLSA(nn.Module):
    """
    Enhanced C3k2 with Global-Local Sparse Attention
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1) 
        
        self.m = nn.ModuleList(
            Bottleneck_GLSA(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        # 严格遵循 CSP 结构：Split -> Bottleneck -> Concat
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

