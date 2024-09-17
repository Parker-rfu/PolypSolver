import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class Mlp(nn.Module):
    """
    This class defines the Feed Forward Network (Multilayer perceptron)
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
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class Self_Attention(nn.Module):
    def __init__(self, dim, ratio_h=2, ratio_w=2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """This class defines the self-attention utilized in the Efficient Transformer block used in the global branch of the G module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            ratio_h (int, optional): the reduction ratio of the height used in efficient transformer block. Defaults to 2.
            ratio_w (int, optional): the reduction ratio of the width used in efficient transformer block.. Defaults to 2.
            num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
            qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
            qk_scale (_type_, optional): Defaults to None.
            attn_drop (_type_, optional): Defaults to 0..
            proj_drop (_type_, optional): Defaults to 0..
        """
        super().__init__()
        self.s = int(ratio_h * ratio_w)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.ke = nn.Conv2d(dim, dim, kernel_size=(ratio_h, ratio_w), stride=(ratio_h, ratio_w), bias=qkv_bias)
        self.ve = nn.Conv2d(dim, dim, kernel_size=(ratio_h, ratio_w), stride=(ratio_h, ratio_w), bias=qkv_bias)
        self.norm_k = nn.LayerNorm(head_dim)
        self.norm_v = nn.LayerNorm(head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.ke(k.transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2).reshape(B, N // self.s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.ve(v.transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2).reshape(B, N // self.s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.norm_k(k)
        v = self.norm_v(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
    
class ETransformer_block(nn.Module):

    def __init__(self, dim, ratio_h=2, ratio_w=2, num_heads=8, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, out_features=None, mlp_ratio=4.,):
        """This class defines the Efficient Transformer block used in the global branch of the G module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            ratio_h (int, optional): the reduction ratio of the height used in efficient transformer block. Defaults to 2.
            ratio_w (int, optional): the reduction ratio of the width used in efficient transformer block.. Defaults to 2.
            num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
            qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
            qk_scale (_type_, optional): Defaults to None.
            attn_drop (_type_, optional): Defaults to 0..
            proj_drop (_type_, optional): Defaults to 0..
            act_layer (_type_, optional): the action function used in FFN. Defaults to nn.GELU.
            norm_layer (_type_, optional): Defaults to nn.LayerNorm.
            out_features (_type_, optional): Defaults to None.
            mlp_ratio (_type_, optional): Defaults to 4..
        """
        super().__init__()
        self.out_features = out_features
        self.norm1 = norm_layer(dim)
        self.attn = Self_Attention(
            dim, ratio_h, ratio_w, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_features, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.out_features:
            return self.mlp(self.norm2(x))
        else:
            return x + self.mlp(self.norm2(x))    
    
    
class Self_Attention_local(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """This class defines the self-attention utilized in the Efficient Transformer block used in the local branch of the G module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
            qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
            qk_scale (_type_, optional): Defaults to None.
            attn_drop (_type_, optional): Defaults to 0..
            proj_drop (_type_, optional): Defaults to 0..
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, R, N, C = x.shape
        qkv = self.qkv(x).reshape(B, R, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(-1, -2).reshape(B, R, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class ETransformer_block_local(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, num_heads=8, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, out_features=None, mlp_ratio=4.,):
        """This class defines the Efficient Transformer block used in the local branch of the G module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
            qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
            qk_scale (_type_, optional): Defaults to None.
            attn_drop (_type_, optional): Defaults to 0..
            drop (_type_, optional): Defaults to 0..
            act_layer (_type_, optional): Defaults to nn.GELU.
            norm_layer (_type_, optional): Defaults to nn.LayerNorm.
            out_features (_type_, optional): Defaults to None.
            mlp_ratio (_type_, optional): Defaults to 4..
        """
        super().__init__()
        self.out_features = out_features
        self.norm1 = norm_layer(dim)
        self.attn = Self_Attention_local(
            dim, qkv_bias=qkv_bias, qk_scale=qk_scale, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_features, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.out_features:
            return self.mlp(self.norm2(x))
        else:
            return x + self.mlp(self.norm2(x))
    
    
class G(nn.Module):
    def __init__(self, dim, ph=4, pw=4, ratio_h=2, ratio_w=2, num_heads=8, drop=0., attn_drop=0.):
        """this class defines the Multiscale Efficient Transformer Attention module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            ph (int, optional): the patch size of height in the local branch. Defaults to 4.
            pw (int, optional): the patch size of width in the local branch. Defaults to 4.
            ratio_h (int, optional): the reduction ratio of the height used in efficient transformer block. Defaults to 2.
            ratio_w (int, optional): the reduction ratio of the width used in efficient transformer block.. Defaults to 2.
            num_heads (int, optional): Defaults to 8.
            drop (_type_, optional): Defaults to 0..
            attn_drop (_type_, optional): Defaults to 0..
        """
        super().__init__()
        self.ph = ph
        self.pw = pw
        self.loc_attn = ETransformer_block_local(dim=dim, num_heads=num_heads, drop=drop, attn_drop=attn_drop)
        self.glo_attn = ETransformer_block(dim=dim, ratio_h=ratio_h, ratio_w=ratio_w, num_heads=num_heads, drop=drop, attn_drop=attn_drop)

    def forward(self, x, feature=False):
        b, c, h, w = x.shape
        loc_x = rearrange(x, 'b d (nh ph) (nw pw) -> b (nh nw) (ph pw) d', ph=self.ph, pw=self.pw)
        glo_x = x.flatten(2).transpose(1, 2)
        loc_y = self.loc_attn(loc_x)
        loc_y = rearrange(loc_y, 'b (nh nw) (ph pw) d -> b d (nh ph) (nw pw)', nh=h // self.ph, nw=w // self.pw,
                          ph=self.ph, pw=self.pw)
        glo_y = self.glo_attn(glo_x)
        glo_y = glo_y.transpose(1, 2).reshape(b, c, h, w)
        y = loc_y + glo_y
        # y = glo_y
        y = torch.sigmoid(y)
        if feature:
            return loc_y, glo_y, x * y
        else:
            return x * y
