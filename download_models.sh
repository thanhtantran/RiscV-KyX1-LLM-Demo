#!/bin/bash

# Yêu cầu người dùng nhập FILE_ID
read -p "Nhập FILE_ID của Google Drive: " FILE_ID

# Tên file nén
FILENAME="file.tar.gz"
TARGET_DIR="models"

# Tải file
echo "🔽 Đang tải file..."
gdown "$FILE_ID" -O "$FILENAME"

# Kiểm tra tải thành công
if [ ! -f "$FILENAME" ]; then
    echo "❌ Tải file thất bại. Thoát."
    exit 1
fi

# Tạo thư mục đích nếu chưa tồn tại
mkdir -p "$TARGET_DIR"

# Giải nén vào thư mục models
echo "📦 Đang giải nén vào thư mục '$TARGET_DIR'..."
if tar -xzf "$FILENAME" -C "$TARGET_DIR"; then
    echo "✅ Giải nén thành công."
    rm "$FILENAME"
    echo "🗑️ Đã xóa file nén."
else
    echo "❌ Giải nén thất bại. File nén được giữ lại."
fi
