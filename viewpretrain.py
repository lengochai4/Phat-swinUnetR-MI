import torch

# 1. Đường dẫn tới file của bạn
path = './pretrained_ckpt/swin_tiny_patch4_window7_224.pth'

# 2. Load file lên (dùng cpu cho nhẹ)
ckpt = torch.load(path, map_location='cpu')

print("-" * 30)
print(f"LOẠI DỮ LIỆU: {type(ckpt)}")

# 3. Kiểm tra xem file chứa state_dict trực tiếp hay đóng gói trong dict khác
if isinstance(ckpt, dict):
    print("CÁC KEY TRONG FILE:", ckpt.keys())
    
    # Thường trọng số sẽ nằm trong key 'model' hoặc 'state_dict'
    state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
else:
    state_dict = ckpt

print("-" * 30)
print(f"TỔNG SỐ LỚP (LAYERS): {len(state_dict)}")
print("DANH SÁCH 10 LỚP ĐẦU TIÊN VÀ KÍCH THƯỚC:")

# 4. Liệt kê tên lớp và kích thước tensor trọng số
for i, (name, param) in enumerate(state_dict.items()):
    if i < 15: # In ra 15 lớp đầu để kiểm tra
        print(f"{i+1}. {name: <40} | Shape: {list(param.shape)}")

# 5. Kiểm tra riêng lớp đầu tiên (Patch Embedding) mà bạn đang quan tâm
print("-" * 30)
key_target = "patch_embed.proj.weight"
if key_target in state_dict:
    print(f"TRỌNG SỐ LỚP ĐẦU ({key_target}):")
    print(f"Shape hiện tại: {state_dict[key_target].shape} (C_out, C_in, H, W)")
else:
    # Nếu không tìm thấy, có thể do tiền tố (prefix) khác nhau, ta tìm kiếm gần đúng
    for k in state_dict.keys():
        if "proj.weight" in k:
            print(f"Gợi ý lớp tương tự: {k} | Shape: {list(state_dict[k].shape)}")
            break