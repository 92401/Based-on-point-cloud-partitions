#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import logging
import os
import copy
from glob import glob
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, PartitionScene
from scene.ptgs.appearance_network import decouple_appearance
from utils.general_utils import safe_state
from utils.partition_utils import data_partition, read_camList
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.manhattan_utils import get_man_trans
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import multiprocessing as mp
from seamless_merging import seamless_merge
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

#主函数
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    #数据集，优化器配置参数，流水线参数，模型在训练过程中需要进行测试的迭代次数，模型的保存间隔迭代次数，设置检查点的迭代次数列表，初始化时加载的检查点路径，调试的起始迭代点。
    first_iter = 0 #初始化迭代次数变量。
    tb_writer = prepare_output_and_logger(dataset)  #设置 TensorBoard 写入器和日志记录器，这两个记录训练过程中的数据和日志信息。此函数在本文件中
    gaussians = GaussianModel(dataset.sh_degree) #实例化高斯模型（重点看，需要转跳）创建一个 GaussianModel 类的实例，输入一系列参数，其参数取自数据集。
    #Gaussian_model调用入口
    scene = Scene(dataset, gaussians) #（这个类的主要目的是处理场景的初始化、保存和获取相机信息等任务，）创建一个 Scene类的实例，使用数据集和之前创建的 GaussianModel 实例作为参数。在scene文件中
    #scene模块入口
    gaussians.training_setup(opt) #设置 GaussianModel 的训练参数。函数在gaussian_model中
    if checkpoint: #如果有提供检查点路径。
        (model_params, first_iter) = torch.load(checkpoint)#通过 torch.load(checkpoint) 加载检查点的模型参数和起始迭代次数。若有检查点first_iter已经被改变
        gaussians.restore(model_params, opt)#通过 gaussian.restore 恢复模型的状态。gaussian_model的方法

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0] #设置背景颜色，根据数据集是否有白色背景来选择。
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") #将背景颜色转化为 PyTorch Tensor，并移到 GPU 上。

    # 创建两个 CUDA 事件，用于测量迭代时间。
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None  #存储训练相机的列表
    ema_loss_for_log = 0.0  #存储损失的指数移动平均值，
    # 在训练过程中，这个值会逐步更新，以便平滑损失的变化情况，使得在日志或进度条中显示的损失更稳定，更能反映总体趋势，而不是过于波动的即时损失值。
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress") #创建一个 tqdm 进度条，用于显示训练进度。
    first_iter += 1  #更新迭代次数，确保不会重复上一次训练的最后一次迭代
    # 接下来开始循环迭代
    for iteration in range(first_iter, opt.iterations + 1): #主要的训练循环开始。从这里开始一直到此函数结束，iteration是一次循环的数
        if network_gui.conn == None: #检查 GUI 是否连接，如果连接则接收 GUI 发送的消息。
            network_gui.try_connect()
        while network_gui.conn != None:#连上了之后
            try:
                net_image_bytes = None #存储生成的图像数据
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()  #recive函数在network_gui中
                #上句代码从GUI界面获取数据，custom_cam：用户自定义的相机参数，do_training：指示是否继续训练的布尔值
                #pipe.convert_SHs_python 和 pipe.compute_cov3D_python：可能是用于控制计算细节的布尔变量
                    #keep_alive：指示是否保持连接的布尔值
                    #scaling_modifer：缩放参数，影响图像生成的比例
                if custom_cam != None: #如果用户在指定了相机参数，则进入图像渲染
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]  #gaussian_render的文件夹下的init文件
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())  #存储图像数据
                network_gui.send(net_image_bytes, dataset.source_path)#将渲染的图像发送到GUI
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break  #如果设置了继续训练和没达到最大值则和没有保持连接，跳出while
            except Exception as e: #如果有异常则
                network_gui.conn = None

        iter_start.record() #用于测量迭代时间开始。
        gaussians.update_learning_rate(iteration) #更新学习率。

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree() #每 1000 次迭代，增加球谐函数的阶数。

        # Pick a random Camera （随机选择一个训练相机。）
        if not viewpoint_stack: #如果相机栈不为空
            viewpoint_stack = scene.getTrainCameras().copy()  #复制一份相机栈，实现在sence的init文件中
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) #随机取出一份相机

        # Render （渲染图像，计算损失（L1 loss 和 SSIM loss））
        #debug_from的作用，检查迭代次数是否达到设定值，达到启用调试模式
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background #这句原理没看懂
        #设置背景颜色

        #从上一步中的render_pkg提取 image（渲染的图像）、viewspace_point_tensor（视图空间点数据）、visibility_filter（可见性过滤器，例如视锥体剔除器）和 radii（高斯半径数据）。
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
        # decouple appearance model
        decouple_image, transformation_map = decouple_appearance(image, gaussians, viewpoint_cam.uid)
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # Loss
        # Ll1 = l1_loss(image, gt_image)
        Ll1 = l1_loss(decouple_image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        iter_end.record()
        with torch.no_grad(): #记录损失的指数移动平均值，并定期更新进度条。
            #PyTorch 中用于控制梯度计算的上下文管理器。它的主要作用是在其作用域内禁用梯度跟踪，从而减少内存使用并加快计算速度。
            #禁用梯度计算以提高性能，减少内存消耗，用于损失记录和进度条更新
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            #计算损失的指数移动平均值（EMA）,就是上面提到的那个参数
            if iteration % 10 == 0: #每10次更新进度条显示的EMA信息
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)  #进度条更新10
            if iteration == opt.iterations: #迭代结束关闭进度条
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            #调用函数记录当前训练状态，包括损失值和时间消耗，将结果写入 TensorBoard 或其他日志系统，具体在本文件中
            if (iteration in saving_iterations): #如果达到保存迭代次数，保存场景。
                print("\n[ITER {}] Saving Gaussians".format(iteration)) #输出当前保存进度
                scene.save(iteration)  #保存

            # Densification（在一定的迭代次数内进行密集化处理。）
            if iteration < opt.densify_until_iter: #在达到指定的迭代次数（密集化截止数）之前执行以下操作。
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #将每个像素位置上的最大半径记录在 max_radii2D 中。这是为了密集化时进行修剪（pruning）操作时的参考。
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                #将与密集化相关的统计信息添加到 gaussians 模型中，包括视图空间点和可见性过滤器。
                if gaussians._xyz.shape[0]<6000000:
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0: #在指定的迭代次数（密集化起始点）之后，每隔一定的迭代间隔进行以下密集化操作。
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None #根据当前迭代次数设置密集化的阈值。如果当前迭代次数大于 opt.opacity_reset_interval，则设置 size_threshold 为 20，否则为 None。
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold) #执行密集化和修剪操作，其中包括梯度阈值、密集化阈值、相机范围和之前计算的 size_threshold。
                
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter): #在每隔一定迭代次数或在白色背景数据集上的指定迭代次数时，执行以下操作。
                        gaussians.reset_opacity() #重置模型中的某些参数，涉及到透明度的操作，具体实现可以在 reset_opacity 方法中找到。

            # Optimizer step（执行优化器的步骤，然后清零梯度。）
            if iteration < opt.iterations:  #迭代次数未达到最大值
                gaussians.optimizer.step()  #优化器更新
                gaussians.optimizer.zero_grad(set_to_none = True) #清空梯度

            # 如果达到检查点迭代次数，保存检查点。
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))  #输出检查点状态
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    #这个函数是用来设置输出文件夹和日志记录系统，输入数据集返回一个日志记录的TensorBoard和创建记录参数的目录
    #这段代码的目的是在用户未指定模型输出路径的情况下，动态生成一个唯一的输出路径。
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))  #args 中的所有参数以字符串形式写入 cfg_log_f 文件

    # Create Tensorboard writer
    tb_writer = None  #初始化TensorBoard 写入器
    if TENSORBOARD_FOUND: #判断上面那个是否可用
        tb_writer = SummaryWriter(args.model_path) # SummaryWriter是 tb_writer的一个类，TensorBoard 是一个可视化工具，能够展示 SummaryWriter 记录的数据
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
# tb_writer：TensorBoard 的写入器，用于记录训练过程中的数据。
# iteration：当前的迭代次数。
# Ll1：L1损失的值，通常表示模型输出与真实值之间的绝对差异。
# loss：总体损失，通常是训练过程中使用的主要损失函数值。
# l1_loss：用于计算 L1 损失的函数。
# elapsed：迭代所花费的时间。
# testing_iterations：在这些特定的迭代次数上进行模型测试。
# scene：场景对象，包含与渲染相关的信息。
# renderFunc：渲染函数，用于生成图像。
# renderArgs：渲染函数的额外参数。/函数负责在训练过程中记录重要的损失和性能指标，并在特定迭代中进行评估。通过将数据记录到 TensorBoard，用户可以直观地监控模型的训练过程和性能，同时通过控制台输出获得实时反馈
    if tb_writer: #将 L1 loss、总体 loss 和迭代时间写入 TensorBoard。
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 在指定的测试迭代次数，进行渲染并计算 L1 loss 和 PSNR。
    # Report test and samples of training set
    if iteration in testing_iterations:  #当前迭代次数在 testing_iterations 列表中，则进行模型测试。
        torch.cuda.empty_cache()  #CUDA 缓存，释放 GPU 内存
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        #定义两个验证配置：一个用于测试集（test），一个用于训练集（train）  两个字典组成的元组

        for config in validation_configs:   #config是一个字典
            if config['cameras'] and len(config['cameras']) > 0:  #如果摄像机列表非空，则继续处理
                #初始化 L1 损失和 PSNR（峰值信噪比）
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):  #相机的索引和视点
                    # 获取渲染结果和真实图像
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):  # tb_writer存在并且当前视角索引小于 5，在 TensorBoard 中记录渲染结果和真实图像
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                     # 计算 L1 loss 和 PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                 # 计算平均 L1 loss 和 PSNR
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])  
                # 在控制台打印评估结果        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                # 在 TensorBoard 中记录评估结果
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # 在 TensorBoard 中记录场景的不透明度直方图和总点数。
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()#使用 torch.cuda.empty_cache() 清理 GPU 内存。

if __name__ == "__main__":  #主函数
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")  #参数解析器对象
    #模型、优化和流水线参数，具体内容在参数声明文件argument
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    #命令行参数，包括 IP 地址、端口、调试选项、测试迭代次数、保存迭代次数等。
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")  #该参数用于控制程序是否在训练过程中输出详细信息
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])#指定程序在训练过程中生成检查点的具体迭代次数
    parser.add_argument("--start_checkpoint", type=str, default = None)#训练开始时加载的检查点文件路径
    #自定义保存的迭代次数
    args = parser.parse_args(sys.argv[1:])#解析命令行参数
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port) #这行代码初始化一个 GUI 服务器，使用 args.ip 和 args.port 作为参数。一个用于监视和控制训练过程的图形用户界面的一部分。
    torch.autograd.set_detect_anomaly(args.detect_anomaly) #设置 PyTorch 是否要检测梯度计算中的异常。
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    # 输入的参数包括：模型的参数（传入的为数据集的位置）、优化器的参数、其他pipeline的参数，测试迭代次数、保存迭代次数 、检查点迭代次数 、开始检查点 、调试起点

    # All done
    print("\nTraining complete.")
