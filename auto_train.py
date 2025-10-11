import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="自动训练所有子区域的模型")
    parser.add_argument('--base_path', type=str, required=False,
                        default=r"E:\model\partition_point_cloud\visible",
                        help='包含所有子区域数据目录的路径,默认 E:\\model\\partition_point_cloud\\visible')
    parser.add_argument('--train_script', type=str, required=False, default="simple_trainer.py",
                        help='训练模型的脚本文件名,默认为 train.py')
    args = parser.parse_args()

    base_path = args.base_path
    train_script = args.train_script

    # 获取 base_path 下的所有子目录，例如：visible下的 01,02,03 ...
    sub_dirs = [d for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))]

    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_path, sub_dir)

        # 在子目录下寻找 partition_xx 目录
        partition_path = None
        for f in os.listdir(sub_dir_path):
            full_path = os.path.join(sub_dir_path, f)
            # 只处理文件夹，并且文件夹名以 "partition_" 开头
            if os.path.isdir(full_path) and f.startswith("partition_"):
                partition_path = full_path
                break
        if partition_path is None:
            print(f"子目录 {sub_dir_path} 下未找到 'partition_xx' 目录，跳过该子目录。")
            continue

        # 输出目录为 partition_xx\output (训练脚本会自动创建)
        output_path = os.path.join(partition_path, "output")

        print(f"正在对目录 {partition_path} 进行模型训练...")

        # 训练命令：python train.py -s inputpath -m outputpath --data_device cpu
        # inputpath: partition_path
        # outputpath: partition_path\output
        cmd = f"CUDA_VISIBLE_DEVICES = 0 python {train_script} brush --data_dir \"{partition_path}\" --data_factor 1 --result_dir \"{output_path}\" "

        try:
            subprocess.check_call(cmd, shell=True)
            print(f"目录 {partition_path} 的模型训练完成！\n")
        except subprocess.CalledProcessError as e:
            print(f"训练目录 {partition_path} 的模型时出现错误：{e}\n")

    print("所有子目录的训练任务已完成！")


if __name__ == "__main__":
    main()

