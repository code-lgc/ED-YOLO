


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ECA(nn.Module):
    """
    高效通道注意力模块 (Efficient Channel Attention)
    用于筛选有效的边缘特征，抑制背景噪声。
    """
    def __init__(self, channels, kernel_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 动态调整 kernel size，或者固定为3/5，这里为了适应性计算k
        t = int(abs((math.log(channels, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

# ================= 核心：软启动差分卷积 =================
class SoftDifferenceConv2d(nn.Module):
    """
    改进版差分卷积：
    1. 加入初始化控制，避免训练初期梯度爆炸或AP骤降。
    2. 显式结合中心差分和局部特征。
    """
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, g=1, act=True):
        super().__init__()
        self.kernel_size = k
        self.stride = s
        self.padding = p
        self.groups = g

        # 权重
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // g, k, k))
        # theta 参数：控制差分项的权重
        self.theta = nn.Parameter(torch.zeros(1, out_channels, 1, 1)) 
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        # 关键：初始化 theta 为一个很小的值（如 0.1 或 0），让它从标准卷积开始学习
        # 这样不会在初期破坏预训练的 Backbone 特征
        nn.init.constant_(self.theta, 0.0) 

    def forward(self, x):
        # 1. 标准卷积项
        out_normal = F.conv2d(x, self.weight, None, self.stride, self.padding, 1, self.groups)
        
        # 2. 差分卷积项
        kernel_diff = self.weight.sum((2, 3), keepdim=True)
        if self.stride > 1:
            x_down = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        else:
            x_down = x
        out_diff = F.conv2d(x_down, kernel_diff, None, 1, 0, 1, self.groups)
        
        # 3. 融合：Out = Normal - Theta * Diff
        # 随着 theta 学习，网络会自动决定哪里需要加强边缘
        return self.act(self.bn(out_normal - self.theta * out_diff))

# ================= 瓶颈层：双路融合 =================
class EdgeSelectBottleneck(nn.Module):
    """
    选择性边缘引导瓶颈层
    Branch 1: 3x3 SoftDifferenceConv (提取边缘)
    Branch 2: 3x3 Standard Conv (提取语义/纹理)
    Fusion: Concat -> Conv -> ECA Attention
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        
        # 提取 k
        k_size = k[0] if isinstance(k, (list, tuple)) else k
        
        # 分支1：语义分支 (Standard)
        self.cv_semantic = Conv(c_, c_, k=k_size, s=1, g=g)
        
        # 分支2：边缘分支 (Difference) - 专门捕捉 BDD 中的车道线、车辆轮廓
        self.cv_edge = SoftDifferenceConv2d(c_, c_, k=k_size, s=1, p=k_size//2, g=g)
        
        # 融合层
        self.fusion = Conv(2 * c_, c2, 1, 1) 
        
        # 注意力筛选：融合后加一个 ECA 模块，抑制噪声
        self.attn = ECA(c2)
        
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        
        # 并行计算
        feat_sem = self.cv_semantic(y)
        feat_edge = self.cv_edge(y)
        
        # 拼接融合
        z = torch.cat((feat_sem, feat_edge), dim=1)
        z = self.fusion(z)
        
        # 注意力加权
        z = self.attn(z)
        
        return x + z if self.add else z

# ================= 最终模块：C3k2_EdgeSelect =================
class C3k2_EdgeSelect(nn.Module):
    """
    用于替换 YOLO11 Backbone 中的 C3k2
    使用 EdgeSelectBottleneck 替代标准 Bottleneck
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        # 初始化 n 个 EdgeSelectBottleneck
        # 这里强制使用 kernel=3 来捕获局部边缘
        self.m = nn.ModuleList(
            EdgeSelectBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        self.cv3 = Conv(2 * self.c, c2, 1)

    def forward(self, x):
        # 仿照 C3k2 结构
        y = self.cv1(x)
        for m in self.m:
            y = m(y)
        return self.cv3(torch.cat((y, self.cv2(x)), 1))