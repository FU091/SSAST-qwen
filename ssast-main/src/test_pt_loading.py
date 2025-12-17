# test_pt_loading.py
import torch
import json
import os
import re
import traceback

def test_pt_file_loading():
    print("=== 測試 .pt 檔案讀取 ===")

    # 測試從 JSON 獲取檔名，然後查找對應的 .pt 檔案
    json_path = r"C:\Users\Lin\Desktop\2_code\ssast_hub\combined_train_data.json"

    if not os.path.exists(json_path):
        print(f"❌ JSON 文件不存在: {json_path}")
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)

        print(f"✅ JSON 文件載入成功，包含 {len(data_json['data'])} 個項目")

        # 測試第一個項目
        if len(data_json['data']) > 0:
            first_item = data_json['data'][0]
            original_path = first_item['wav']
            print(f"第一個項目原始路徑: {original_path}")

            # 提取檔名
            path_parts = re.split(r'[\\/]+', original_path)
            filename = path_parts[-1]
            filename_no_ext = os.path.splitext(filename)[0]
            print(f"提取的檔名: {filename_no_ext}")

            # 構造 .pt 檔案路徑
            pt_dir = r"D:\spectrogram_pt_name"
            pt_path = os.path.join(pt_dir, f"{filename_no_ext}.pt")
            print(f"構造的 .pt 路徑: {pt_path}")

            if os.path.exists(pt_path):
                print("✅ .pt 檔案存在")

                # 測試載入 .pt 檔案
                try:
                    data_dict = torch.load(pt_path)
                    print(f"✅ .pt 檔案載入成功")

                    if 'x' in data_dict and 'y' in data_dict:
                        print(f"✅ .pt 檔案格式正確 - x shape: {data_dict['x'].shape}, y shape: {data_dict['y'].shape}")
                        print(f"✅ 資料類型 - x: {data_dict['x'].dtype}, y: {data_dict['y'].dtype}")
                        return True
                    else:
                        print("❌ .pt 檔案格式錯誤，缺少 x 或 y 鍵")
                        return False

                except Exception as e:
                    print(f"❌ .pt 檔案載入錯誤: {e}")
                    return False
            else:
                print(f"❌ 找不到對應的 .pt 檔案: {pt_path}")
                print("可能的 .pt 檔案列表:")
                if os.path.exists(pt_dir):
                    pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
                    if pt_files:
                        for i, pt_file in enumerate(pt_files[:5]):  # 只顯示前5個
                            print(f"  - {pt_file}")
                    else:
                        print("  - 資料夾內沒有找到 .pt 檔案")
                else:
                    print(f"  - 目標目錄不存在: {pt_dir}")
                return False
        else:
            print("❌ JSON 資料中沒有 'data' 項目")
            return False

    except Exception as e:
        print(f"❌ JSON 載入錯誤: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pt_file_loading()
    if success:
        print("\n🎉 .pt 檔案讀取測試成功！")
    else:
        print("\n💥 .pt 檔案讀取測試失敗！")