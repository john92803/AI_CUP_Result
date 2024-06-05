
import os
import shutil

def copy_txt_files(source_dir, dest_dir):

    if not os.path.isdir(source_dir):
        raise ValueError(f"來源資料夾 '{source_dir}' 不存在")

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    for root, _, files in os.walk(source_dir):
        for root1, _, files in os.walk(root):
            for filename in files:
                if filename.endswith(".txt"):
                    src_file = os.path.join(root1, filename)
                    dst_file = os.path.join(dest_dir, filename)
                    shutil.copyfile(src_file, dst_file)

if __name__ == "__main__":
    source_dir = "/home/user/pt_aicup/AICUP_Baseline_BoT-SORT/runs/detect"
    dest_dir = "/home/user/pt_aicup/AICUP_Baseline_BoT-SORT/runs/test_res"

    copy_txt_files(source_dir, dest_dir)
