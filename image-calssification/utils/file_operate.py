import os
import shutil

from utils.data_loader import get_project_root
import torch


def copy_file(src, dst):
    source_file_path = src
    destination_folder_path = dst
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    # 复制文件
    shutil.copy(source_file_path, destination_folder_path)


def save_image(image_path, dst):
    path = get_project_root() + image_path
    copy_file(path, dst)


def save_model(model, name):
    project_root = get_project_root()
    save_path = str(project_root) + f"/store/{name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"保存模型成功，保存到了{save_path}")


def get_trained_model(path):
    project_root = get_project_root()
    save_path = str(project_root) + f"/store/model/{path}.pth"
    return torch.load(save_path,weights_only=True)
