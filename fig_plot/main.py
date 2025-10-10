import subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 文件路径
fig_files = [
    "E:/code/bigGaussian-refactor/fig_plot/fig3.py",
    "E:/code/bigGaussian-refactor/fig_plot/fig4.py",
    "E:/code/bigGaussian-refactor/fig_plot/fig5.py",
    "E:/code/bigGaussian-refactor/fig_plot/fig6.py"
]

# 保存的图像路径
fig_image_paths = [
    "fig3.png",
    "fig4.png",
    "fig5.png",
    "fig6.png"
]

# 依次执行每个文件并生成图像
for idx, fig_file in enumerate(fig_files, start=1):
    print(f"正在执行第 {idx} 个画图脚本: {fig_file}")
    subprocess.run(["python", fig_file])

print("所有图形绘制完成！")

# 创建一个2x2的子图布局
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# 加载并显示每张图
for idx, ax in enumerate(axs.flat):
    img = mpimg.imread(fig_image_paths[idx])  # 读取图像
    ax.imshow(img)  # 显示图像
    ax.axis('off')  # 去掉坐标轴
    
    # 设置坐标范围（这里可以根据需要调整具体范围）
    
    # 设置标题，并将其放在图片的下方
    ax.set_title(f" {idx + 1}", fontsize=10, y=-0.15)

# 调整布局，避免重叠
plt.tight_layout()

# 保存拼接后的图像
plt.savefig("combined_figure.png", dpi=300)
plt.show()
