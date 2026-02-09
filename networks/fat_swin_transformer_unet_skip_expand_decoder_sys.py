# coding=utf-8
from __future__ import absolute_import, division, print_function
import logging
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_

#=====================================================================
#TO DO LIST
# 1. Gắn class MOEFNGating vào Swin block
#=====================================================================

# Ghi log lại các thông báo, __name__ - tên của file được đặt (chạy trực tiếp = main, chạy từ file = tên file)
logger = logging.getLogger(__name__)

# Chỉ những hàm, class trong danh sách này mới được import sang file khác
__all__ = [
    "SwinTransformerSys",
]


#Thực hiện ý tưởng MoE - CHƯA ÁP DỤNG HÀM VÀO SWIN BLOCK
class MoEFFNGating(nn.Module):
    """
    Optional MoE-FFN block (currently not wired into Swin blocks by default).
    Kept for compatibility with your original sys file.
    """
    def __init__(self, dim, hidden_dim, num_experts):
        super(MoEFFNGating, self).__init__()
        self.gating_network = nn.Linear(dim, dim)   # Quản lý danh sách experts, tính toán % tham gia của mỗi expert
        self.experts = nn.ModuleList([  # Danh sách experts (2 lớp linear và hàm kích hoạt GELU - Gaussian Error Linear Unit)
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(num_experts)
        ])

    # Xử lý dữ liệu
    def forward(self, x):
        weights = self.gating_network(x)    #Tính trọng số (gating); x shape: [B, L, C] (Batch, Tokens, Dim)
        weights = torch.nn.functional.softmax(weights, dim=-1)  # Chia trọng số ra cho từng experts, tổng lại vẫn = 1; [B, L, num_experts]

        outputs = [expert(x) for expert in self.experts]    # Tất cả experts chung 1 input x
        outputs = torch.stack(outputs, dim=0)   # Xếp chồng kết quả lại thành một tensor lớn

        # outputs = (weights.unsqueeze(0) * outputs).sum(dim=0) # Tổng đóng góp của experts, ưu tiên đặc trưng của expert có trọng số cao nhất
        # Tính tổng trọng số: [B, L, C]
        final_output = (weights.transpose(0, 2).unsqueeze(-1) * outputs).sum(dim=0)

        return final_output, weights


# Multi-layer Perceptron - từng pixel tự biến đổi đặc trưng
# Cấu trúc Inverted Bottleneck
# Quy trình Swin-UNet: Attention (Giao tiếp giữa các pixel) -> MLP (Xử lý đặc trưng riêng lẻ)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features # Mặc định kích thước như đầu vào
        hidden_features = hidden_features or in_features    # Mặc định giữ nguyên, Swin Transformer thì x4 in_features (mlp.ratio)
        self.fc1 = nn.Linear(in_features, hidden_features) # Mở rộng số lượng đặc trưng gấp 4 lần
        self.act = act_layer() # Kích hoạt AF GELU học tính chất nonlinear
        self.fc2 = nn.Linear(hidden_features, out_features) # Nén lại kích thước ban đầu / kích thước yêu cầu
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)    # Dropout qua lớp kích hoạt
        x = self.fc2(x)
        x = self.drop(x)    # Dropout lớp linear thứ 2 -> chống overfitting
        return x


# Khác biệt cho với Global Attention của ViT
# Swin Transformer chia ảnh thành các windows, cho các pixel trong 1 window nhìn nhau
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int)
        B: batch size (số lượng ảnh xử lý cùng lúc)
        H,W: chiều cao, rộng của ảnh.
        C: số kênh (dim)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
        -> Output tensor (Tổng số cửa sổ, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C) # Cắt đường, chia ảnh ra.
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # Hoán vị (permute) sắp xếp lại trục -> chỉ số windows trước, chỉ số pixel sau
    # Gom nhóm (view) đưa tất cả windows của tất cả ảnh trong batch thành một chiều
    return windows


# Làm ngược lại với hàm ở trên, gắn lại thành hình dạng ban đầu
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int)
        H (int)
        W (int)

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size)) # Tổng số cửa sổ đang có chia cho số lượng cửa sổ của 1 ảnh (H/window_size) x (W/window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)   # Định hình lại tensor (view)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)  # Sắp xếp lại các mảnh (permute)
    return x


# Trung tâm của Swin Transformer
class WindowAttention(nn.Module):
    r"""
    Window based multi-head self attention (W-MSA) with relative position bias.
    Supports shifted and non-shifted windows.

    Query, Key, Value (QKV) mechanism
    Q: Tìm kiếm đặc điểm gì?
    K: Bản thân có đặc điểm gì?
    V: Thông tin bản thân mang theo là gì?

    Attention Score = Q.K -> pixel A tập trung vào B bao nhiêu phần

    relative_position_index: vị trí tương đối thay vì tọa độ cố định như Transformer
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads  # Chia dim làm nhiều num_heads, mỗi cái đọc một yếu tố khác nhau (màu sắc, đường kẻ, góc)

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # relative position bias table: bảng bias biểu diễn quan hệ giữa các pixel gần nhau cho mô hình
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])

        # torch>=1.10 recommends specifying indexing
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C)
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # kết quả các đầu được gộp lại (transpose và reshape) để có cái nhìn tổng thể nhất.

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Mask cho Shifted Window - che mắt mô hình khỏi tính toán các pixel lạ
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


# Cốt lõi của Swin
# Cơ chế Shifted Window: nhằm cho các pixel ở mép cửa sổ này biết cái bên cạnh đang làm gì
    # Dịch chuyển ảnh sang trái và lên trên theo shifted_size
    # Dùng Attention Mask cho các vùng rìa bị cuộn sang bên khác (nằm xa vị trí gốc) để skip attention
    # Giúp việc tính toán trên các cửa sổ bị dịch chuyển vẫn đảm bảo tính logic về mặt không gian địa lý của ảnh
# Shorcut / skip connection [shortcut]: 2 lần -> sau attention & sau MLP
class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block."""
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_experts=4
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in [0, window_size)"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP cũ: self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # MoE mới:
        self.mlp = MoEFFNGating(
            dim=dim, 
            hidden_dim=mlp_hidden_dim, 
            num_experts=num_experts
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))

            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):   # Luồng xử lý
        # Chuẩn hóa
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        #Shifted Window
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Chia cửa sổ
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Tính Attention, kèm Mask chặn các vùng không liên quan
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.use_moe:
            # Tính toán gating và đầu ra expert
            # gating_weights shape: [B, L, num_experts]
            x_moe, gating_weights = self.mlp(x) 
            x = shortcut + self.drop_path(x_moe)
            
            # Lưu lại trọng số để tính Loss
            if hasattr(self, 'all_gating_weights'):
                self.all_gating_weights.append(gating_weights)
        else:
            x = shortcut + self.drop_path(self.mlp(x))
        return x

    # Hai hàm cho quản lý & debug
    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


# Tạo ra cấu trúc phân tầng (hierachical)
# Giảm /2 độ phân giải, x2 độ sâu đặc trưng
class PatchMerging(nn.Module):
    """Patch Merging Layer."""
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        #Bốc thăm pixel (Slicing) thành 4 nhóm
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]    # Hàng chẵn cột chẵn
        x1 = x[:, 1::2, 0::2, :]    # Hàng lẻ cột chẵn
        x2 = x[:, 0::2, 1::2, :]    # Hàng chẵn cột lẻ
        x3 = x[:, 1::2, 1::2, :]    # Hàng lẻ cột lẻ
        x = torch.cat([x0, x1, x2, x3], -1)  # 4 nhóm xếp chồng lên nhau -> B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)

        x = self.norm(x) # Chuẩn hóa về vùng ổn định
        x = self.reduction(x)   # Nén số kênh 4xC còn 2xC
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


# Khôi phục lại kích thước ảnh sau khi có đặc trưng
class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale

        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)  # Mở rộng kênh để đủ nguyên liệu dàn trải pixel
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(  # Pixel shuffle -> lấy C kênh chia ra lấp đầy vào 4 pixel mới
            # (28, 28, 192) -> (56, 56, 96)
            x,
            "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // (self.dim_scale ** 2),
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        num_experts=4
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2, # Block chẵn -> cố định, Block lẻ -> shifted window
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                num_experts=num_experts
            )
            for i in range(depth)
        ])

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """A basic Swin Transformer layer for one stage (decoder)."""
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        upsample=None,
        use_checkpoint=False,
        num_experts=4
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                num_experts=num_experts
            )
            for i in range(depth)
        ])

        self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer) if upsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


# Biến đổi ảnh màu RGB thành các feature vectors cho mô hình hiểu
# Dịch từ pixel sang token để Transformer làm việc
class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans #Số modal * Số lát cắt
        self.embed_dim = embed_dim

        # Conv2d chia thành các patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, Ph*Pw, C
            # self.proj(x): (B, 3, 224, 224) qua Conv2d -> (B, 96, 56, 56)
            # flatten(2): -> (B, 96, 3136)
            # transpose(1, 2): đưa về dạng input của Transformer (Batch, Số lượng Patch, Số kênh) -> (B, 3136, 96)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


# Class tổng hợp, kết nối tất cả hàm trên -> hệ thống Swin-UNet
class SwinTransformerSys(nn.Module):
    r"""
    Swin-UNet style encoder-decoder with skip connections + final x4 upsample.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=12,    # BraTS có 4 modals, mỗi modal lấy 3 lát cắt (4*3=12)
        num_classes=4,   # BraTS có 3 nhãn u + 1 nhãn nền = 4
        embed_dim=96,
        depths=(2, 2, 2, 2),    # Số lượng Swinblocks nhánh xuống
        depths_decoder=(1, 2, 2, 2),    # Số lượng Swinblocks nhánh lên
        num_heads=(3, 6, 12, 24),   # Tăng dần để quan sát đặc trưng phức tạp ở các kênh sâu
        window_size=7,
        num_experts=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,  # Absolute Position Embedding -> bộ lộc vị trí tuyệt đối ngay từ đầu ảnh
        patch_norm=True,
        use_checkpoint=False,
        final_upsample="expand_first",
        **kwargs
    ):
        super().__init__()

        logger.info(
            "SwinTransformerSys init | depths=%s depths_decoder=%s drop_path_rate=%.4f num_classes=%s",
            list(depths), list(depths_decoder), float(drop_path_rate), str(num_classes)
        )

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.gating_weights_list = []

        # patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution

        # absolute pos embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # encoder
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2 ** i_layer),
                    self.patches_resolution[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                num_experts=num_experts,
            )
            self.layers.append(layer)

        # decoder
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = (
                nn.Linear(
                    2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )

            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(
                        self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer,
                )
            else:
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:(self.num_layers - 1 - i_layer)]):sum(depths[:(self.num_layers - 1 - i_layer) + 1])
                    ],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    num_experts=num_experts,
                )

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            logger.info("Final upsample: expand_first")
            self.up = FinalPatchExpand_X4(
                input_resolution=(img_size // patch_size, img_size // patch_size),
                dim_scale=4,
                dim=embed_dim,
            )
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):  # Nhánh Encoder đi xuống
        x = self.patch_embed(x) # Cắt ảnh thành các miếng 4x4
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_downsample = []   # Lưu một bản sao qua mỗi BasicLayer, cho skip connection
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B, L, C
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        # NOTE: original logic kept (assumes 4 stages; uses x_downsample[3 - inx])
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)   # Skip connection: Lấy dữ liệu phóng to ghép với dữ liệu lưu trữ
                x = self.concat_back_dim[inx](x)    # Đưa về lại kênh trước skip connection (nến /2 kênh lại)
                x = layer_up(x)

        x = self.norm_up(x)  # B, L, C
        return x

    def up_x4(self, x): # Phóng ảnh lên x4 để về 224x224
        H, W = self.patches_resolution
        B, L, C = x.shape
        
        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = self.output(x)  
            
        # Với BraTS, thường dùng Softmax hoặc Sigmoid cuối cùng
        # return torch.softmax(x, dim=1) # Nếu phân loại các lớp tách biệt
        return x # Trả về logits để dùng với CrossEntropyLoss hoặc DiceLoss

    def forward(self, x):
        self.gating_weights_list = []
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        return x, self.gating_weights_list

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
