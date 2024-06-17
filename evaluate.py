import os
import time
import yaml
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from models import ICNet
from dataset.cityscapes import CustomCOCODataset
from utils import SegmentationMetric, SetupLogger, get_color_pallete

class Evaluator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 数据集
        val_dataset = CustomCOCODataset(root=cfg["train"]["coco_root"],
                                        split='val',
                                        base_size=cfg["model"]["base_size"],
                                        crop_size=cfg["model"]["crop_size"])
        self.val_dataloader = data.DataLoader(dataset=val_dataset,
                                              batch_size=cfg["train"]["valid_batch_size"],
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              drop_last=False)

        # 模型
        self.model = ICNet(nclass=val_dataset.NUM_CLASS, backbone='resnet50').to(self.device)
        pretrained_net = torch.load(cfg["test"]["ckpt_path"])
        self.model.load_state_dict(pretrained_net)

        # 评估指标
        self.metric = SegmentationMetric(val_dataset.NUM_CLASS)

        # 图像转换
        self.to_pil = transforms.ToPILImage()

    def eval(self):
        self.metric.reset()
        self.model.eval()

        logger.info("Start validation, Total sample: {:d}".format(len(self.val_dataloader.dataset.image_paths)))
        for i, (image, mask, _) in enumerate(self.val_dataloader):
            image = image.to(self.device)
            mask = mask.to(self.device)

            with torch.no_grad():
                start_time = time.time()
                outputs = self.model(image)
                end_time = time.time()
                step_time = end_time - start_time

            self.metric.update(outputs[0], mask)
            pixAcc, mIoU = self.metric.get()

            logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}, time: {:.3f}s".format(
                i + 1, pixAcc * 100, mIoU * 100, step_time))

            # 处理并保存图像
            image_pil = self.to_pil(image.squeeze(0).cpu())
            prefix = os.path.splitext(os.path.basename(self.val_dataloader.dataset.image_paths[i]))[0]
            image_pil.save(os.path.join(outdir, prefix + '_src.png'))

            # 处理并保存掩码
            mask_pil = get_color_pallete(mask.cpu().numpy().squeeze(), "citys")
            mask_pil.save(os.path.join(outdir, prefix + '_label.png'))

        # 计算平均指标
        average_pixAcc, average_mIoU = self.metric.get()
        logger.info("Evaluate: Average mIoU: {:.3f}, Average pixAcc: {:.3f}".format(average_mIoU, average_pixAcc))

if __name__ == '__main__':
    # 配置文件
    config_path = "./configs/icnet.yaml"
    with open(config_path, "r", encoding='utf-8') as yaml_file:
        cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["train"]["specific_gpu_num"])
    outdir = os.path.join(cfg["train"]["ckpt_dir"], "evaluate_output")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    logger = SetupLogger(name="semantic_segmentation",
                         save_dir=cfg["train"]["ckpt_dir"],
                         distributed_rank=0,
                         filename='{}_{}_evaluate_log.txt'.format(cfg["model"]["name"], cfg["model"]["backbone"]))

    evaluator = Evaluator(cfg)
    evaluator.eval()
