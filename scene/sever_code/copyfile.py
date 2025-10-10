import os
import shutil


def organize_cropped_ply_files(base_path, output_folder):
    """
    将裁剪后的 .ply 文件复制到新文件夹，并以 .pkl 文件的名称命名。

    :param base_path: 输入文件的根目录，包含未裁剪的 .pkl 文件和裁剪后的 .ply 文件
    :param output_folder: 输出文件夹路径，存放复制并重命名的文件
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历 base_path 下的所有子目录
    for sub_dir in os.listdir(base_path):
        sub_dir_path = os.path.join(base_path, sub_dir)

        # 检查子目录是否存在并包含 .pkl 文件
        if not os.path.isdir(sub_dir_path):
            continue

        # 查找 .pkl 文件
        pkl_files = [f for f in os.listdir(sub_dir_path) if f.endswith(".pkl")]
        if not pkl_files:
            print(f"子目录 {sub_dir_path} 中未找到 .pkl 文件，跳过...")
            continue

        # 处理每个 .pkl 文件
        for pkl_file in pkl_files:
            pkl_path = os.path.join(sub_dir_path, pkl_file)
            partition_name = os.path.splitext(pkl_file)[0]  # 获取文件名（不含扩展名）

            # 查找对应的裁剪后的 .ply 文件
            ply_path = os.path.join(sub_dir_path, partition_name, "output", "point_cloud", "iteration_30000",
                                    "cutor_pcd.ply")

            if not os.path.exists(ply_path):
                print(f"裁剪后的 .ply 文件未找到：{ply_path}，跳过...")
                continue

            # 构造新的文件名和路径
            new_file_name = f"{partition_name}.ply"
            new_file_path = os.path.join(output_folder, new_file_name)

            # 复制并重命名文件
            shutil.copy2(ply_path, new_file_path)
            print(f"文件已复制并重命名：{ply_path} -> {new_file_path}")

    print(f"所有文件已处理完成，结果保存在：{output_folder}")


if __name__ == "__main__":
    # 根目录，包含未裁剪的 .pkl 文件和裁剪后的 .ply 文件
    base_path = r"E:\airport_data\test\ychdata\model_all\split_result\visible"

    # 输出文件夹路径
    output_folder = r"E:\airport_data\test\ychdata\model_all\split_result\visible\organized"

    # 执行文件组织操作
    organize_cropped_ply_files(base_path, output_folder)
