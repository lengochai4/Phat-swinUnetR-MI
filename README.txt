- Swin UNet, chuyển đổi từ 2D sang 2.5D (input một cụm lát cắt 2D).

- Bỏ cơ chế tự động ép ảnh về 3 kênh, kiểm tra số kênh input có khớp config (12 kênh) hay không.
    + 4 loại modal (T1, T1Gd, T2, FLAIR) trong BraTS, dùng 3 lát cắt liên tiếp = 12 kênh.

- Thay đổi cách load trọng số của pretrain trước khi nạp vào mô hình:
    + Trong pretrain, mô hình theo hướng ảnh 2D -> RGB 3 kênh.
    + Mô hình hiện tại theo hướng 2.5D -> stack lên 12 kênh.
    -> Thay đổi chỉ số lớp đầu tiên của mô hình = Trung bình cộng trọng số của 3 kênh pre-train rồi lặp lại cho 12 kênh.

- Tích hợp lớp MoE (Mixture of Experts):
    + Mỗi expert có 2 lớp linear và hàm kích hoạt GELU - Gaussian Error Linear Unit.
    + Mỗi expert trong MoE có thể học cách chuyên biệt hóa để nhận diện một loại mô cụ thể.
    + Thêm Load Balancing Loss, tránh tình trạng một vài experts gánh toàn bộ, còn lại bị bỏ trống.