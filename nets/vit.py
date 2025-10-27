"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
class GateNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, num_tasks):
        super(GateNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts * num_tasks)  # 为每个任务选择多个专家
        self.num_experts = num_experts
        self.num_tasks = num_tasks

    def forward(self, x, task_index):
        gate_output = self.fc(x)  # (batch_size, num_experts * num_tasks)
        gate_weights = gate_output.view(x.size(0), self.num_tasks,self.num_experts)  # (batch_size, num_tasks, num_experts)
        gate_weights = F.softmax(gate_weights, dim=-1)  # (batch_size, num_tasks, num_experts)

        return gate_weights
# class GateNetwork(nn.Module):
#     def __init__(self, input_dim, num_experts):
#         super(GateNetwork, self).__init__()
#         self.gate_weights = nn.Parameter(torch.randn(num_experts, input_dim))  # w_e, 可学习参数
#
#     def forward(self, x):
#         """
#         x: 输入特征 [num_tasks, feature_dim]
#         输出: Gate权重 [num_tasks, num_experts]
#         """
#         # x @ w_e^T 计算 Gate Score
#         gate_scores = torch.matmul(x, self.gate_weights.T)  # [num_tasks, num_experts]
#         # Softmax 归一化，得到 Gate Weight
#         gate_weights = F.softmax(gate_scores, dim=1)  # [num_tasks, num_experts]
#         return gate_weights

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExpertNetwork, self).__init__()
        # 每个专家是一个简单的全连接网络
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


# class MixtureOfExperts(nn.Module):
#     def __init__(self, input_dim, output_dim, num_experts, num_tasks, confidence_threshold=0.7):
#         super(MixtureOfExperts, self).__init__()
#         self.num_experts = num_experts
#         self.num_tasks = num_tasks
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.confidence_threshold = confidence_threshold  # 置信度阈值
#
#         self.gate = GateNetwork(input_dim, num_experts, num_tasks)
#         self.experts = nn.ModuleList([ExpertNetwork(input_dim, output_dim) for _ in range(num_experts)])
#
#     def forward(self, x, task_index):
#         # 获取门控网络输出的专家权重
#         gate_weights = self.gate(x, task_index)  # (batch_size, num_experts)
#
#         # 获取每个专家的输出
#         expert_outputs = [expert(x) for expert in self.experts]  # 每个专家的输出列表
#         expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, output_dim)
#
#         # 计算主专家的置信度
#         # primary_expert_prob = gate_weights[:, task_index]  # (batch_size, )
#         # 计算每个样本的主专家置信度，选择权重最大的专家
#         primary_expert_prob, primary_expert_idx = gate_weights.max(dim=-1)  # (batch_size, ), (batch_size, )
#
#         # 判断主专家置信度，低于阈值则启用其他专家
#         low_confidence_mask = primary_expert_prob < self.confidence_threshold  # (batch_size, )
#         task_output = expert_outputs * gate_weights.unsqueeze(-1)  # (batch_size, num_experts, output_dim)
#         task_output_sum = task_output.sum(dim=1)
#
#         # 当主专家置信度高于阈值时，只使用主专家的输出
#         #primary_expert_idx = gate_weights.argmax(dim=-1)  # (batch_size, )
#         primary_expert_output = expert_outputs.gather(1, primary_expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1,self.output_dim))
#         primary_expert_output = primary_expert_output.squeeze(1)  # (batch_size, output_dim)
#
#         # 根据低置信度遮罩，决定是使用多个专家还是主专家输出
#         final_output = torch.where(low_confidence_mask.unsqueeze(-1), task_output_sum, primary_expert_output)

        # return final_output,final_output,final_output,final_output,final_output,final_output,final_output
class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, num_tasks, confidence_threshold=0.7):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.confidence_threshold = confidence_threshold  # 置信度阈值

        # 初始化门控网络
        self.gate = GateNetwork(input_dim, num_experts, num_tasks)

        # 初始化专家网络列表
        self.experts = nn.ModuleList([ExpertNetwork(input_dim, output_dim) for _ in range(num_experts)])

    def forward(self, x, task_index):
        # 获取门控网络输出的专家权重
        gate_weights = self.gate(x,task_index)  # (batch_size, num_tasks, num_experts)

        # 获取每个专家的输出
        expert_outputs = [expert(x) for expert in self.experts]  # 每个专家的输出列表
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, output_dim)

        # 选择当前任务的主专家
        task_gate_weights = gate_weights[:, task_index, :]  # (batch_size, )

        # 当前任务计算每个专家的输出
        task_output = expert_outputs * task_gate_weights.unsqueeze(-1)  # (batch_size, num_experts, output_dim)
        task_output_sum = task_output.sum(dim=1)  # (batch_size, output_dim)

        # 计算主专家置信度
        primary_expert_prob,primary_expert_idx = task_gate_weights.max(dim=-1)  # (batch_size, ), (batch_size, )
        # with open('D:\deeplearning\DETER\logs\primary_expert.txt', 'a') as f:
        #     # 追加数据，确保每次都换行
        #     f.write(f"预测：{gate_weights}")
        #     f.write("\n")
        # 判断主专家的置信度是否低于阈值
        low_confidence_mask = primary_expert_prob < self.confidence_threshold  # (batch_size, )

        # 获取主专家的输出
        task_index = torch.tensor([task_index] * x.size(0), device=x.device)
        primary_expert_output = expert_outputs.gather(1, primary_expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.output_dim))
        primary_expert_output = primary_expert_output.squeeze(1)  # (batch_size, output_dim)

        # 如果主专家置信度低，则使用所有专家的加权输出
        final_output = torch.where(low_confidence_mask.unsqueeze(-1), task_output_sum, primary_expert_output)

        return final_output,final_output,final_output,final_output,final_output,final_output,final_output
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth随机深度) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

# class MultiScaleCrossAttention(nn.Module):
#     def __init__(self, embed_dim):
#         super(MultiScaleCrossAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.query_conv = nn.Conv2d(3, embed_dim, kernel_size=1)
#         self.key_conv = nn.Conv2d(3, embed_dim, kernel_size=1)
#         self.value_conv = nn.Conv2d(3, embed_dim, kernel_size=1)
#         self.attn_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
#
#     def forward(self, roi_features, local_features):
#         # Step 1: 生成 Query, Key, Value 特征
#         Q = self.query_conv(roi_features)  # (batch_size, embed_dim, 224, 224)
#         K = self.key_conv(local_features)  # (batch_size, embed_dim, 224, 224)
#         V = self.value_conv(local_features)  # (batch_size, embed_dim, 224, 224)
#
#         # Step 2: 计算注意力权重
#         attn_weights = torch.matmul(Q.flatten(2).transpose(1, 2), K.flatten(2))  # (batch_size, embed_dim, 224, 224)
#         attn_weights = F.softmax(attn_weights, dim=-1)  # (batch_size, embed_dim, 224, 224)
#
#         # Step 3: 计算加权和
#         output = torch.matmul(attn_weights, V.flatten(2))  # (batch_size, embed_dim, 224, 224)
#         output = output.view(-1, self.embed_dim, 224, 224)  # Reshape back to (batch_size, embed_dim, 224, 224)
#
#         # Step 4: 融合特征
#         fused_output = self.attn_conv(output)  # (batch_size, 3, 224, 224)
#
#         return fused_output
class MultiScaleCrossAttention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(MultiScaleCrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.query_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  # 拼接后需要再进行全连接
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self, x,tezheng):
        B, N, C = x.shape
        Q = self.query_conv(x)  # ()
        K = self.key_conv(tezheng)  # ()
        V = self.value_conv(tezheng)  # ()
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ V).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  #直接使用 一个全连接层得到qkv
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  #拼接后需要再进行全连接
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)  #激活函数
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossattentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(CrossattentionBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiScaleCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x,tezheng):
        x = x + self.drop_path(self.attn(self.norm1(x),tezheng))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,  #representation_size  对应MLP  distilled不用管
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1  #不用管 为1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  #构建一个可训练参数
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None #不用管
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) #
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule 构建一个等差序列
        self.crossattentionblock = CrossattentionBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=0,
                  norm_layer=norm_layer, act_layer=act_layer)

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)   #构建了一个列表循环， drop_path_ratio递增
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:  #不用管
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        self.linear_layer = nn.Linear(256,768)
        # Classifier head(s)
        # self.head1 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head2 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head3 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head4 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head5 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head6 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head7 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head = MixtureOfExperts(self.num_features, num_classes, 5, 1, confidence_threshold=0.7)

        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x,tezheng):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        # tezheng = self.patch_embed(tezheng)  #区域特征图 待修改尺寸
        # cls_token = self.cls_token.expand(tezheng.shape[0], -1, -1)
        # tezheng = torch.cat((cls_token, tezheng), dim=1)  # [B, 197, 768]
        # tezheng = self.pos_drop(tezheng + self.pos_embed)
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        tezheng =tezheng.view(tezheng.size(0), tezheng.size(1), -1)
        #tezheng = self.linear_layer(tezheng) #16,512,768  ################################################推理的时候给注释了  训练打开


        x = self.patch_embed(x)  # [B, 196, 768]
       # print(x.shape)
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)
        #########插入交叉注意力
        #x = self.crossattentionblock(x,tezheng)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])   #不做任何操作
            #return torch.mean(x, dim=1)
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x,tezheng,task_index):
        x = self.forward_features(x,tezheng)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x1, x2, x3, x4, x5, x6, x7 = self.head(x,task_index)
            # x1 = self.head1(x)
            # x2 = self.head2(x)
            # x3 = self.head3(x)
            # x4 = self.head4(x)
            # x5 = self.head5(x)
            # x6 = self.head6(x)
            # x7 = self.head7(x)
        return x1,x2,x3,x4,x5,x6,x7


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model

# net = vit_base_patch16_224()
# task_index = 1
# x = torch.Tensor(1,3,224,224)
# tezheng = torch.Tensor(1,3,224,224)
# net(x,tezheng,task_index)



# input_dim = 768
# output_dim = 8
# num_experts = 7
# num_tasks = 1
# model = MixtureOfExperts(input_dim, output_dim, num_experts, num_tasks, confidence_threshold=0.7)
#
# x = torch.randn(16, 768)  # (batch_size, input_dim)
# task_index = 0  # 当前任务索引
# output = model(x, task_index)
#
# print(output.shape)