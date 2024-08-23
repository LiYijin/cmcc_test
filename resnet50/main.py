import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import onnxruntime as ort
import onnxruntime.capi as ort_cap
import numpy as np
import random
from torchvision.models import ResNet50_Weights
import argparse
import time
import multiprocessing
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--precision', '-P',choices=['fp32', 'fp16'], help='Specify precision mode (fp32 or fp16)', required=True)
parser.add_argument('--gpu_id', '-id', help='Specify gpu id', required=True)

args = parser.parse_args()


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    transforms.CenterCrop(32),
])

# 加载训练集
infer_dataset = CIFAR100(root='./dataset', train=False, download=True, transform=transform)
infer_dataset = DataLoader(dataset=infer_dataset, batch_size=24, shuffle=False)

resnet_test = ort.InferenceSession("./resnet-{}.onnx".format(args.precision),
                                   providers=[('MUSAExecutionProvider', {"prefer_nhwc": '1'})])
# model = models.resnet50()
# model_path = './resnet50_b16x8_cifar100_20210528-67b58a1b.pth'
# checkpoint = torch.load(model_path)['state_dict']
# # 加载状态字典到模型
# # print(checkpoint)
# adjusted_state_dict = {}
# for k, v in checkpoint.items():
#     if k.startswith('backbone.'):
#         k = k[9:]
#     adjusted_state_dict[k] = v
# print(adjusted_state_dict)
# model.load_state_dict(adjusted_state_dict)
# model.eval()
def evaluate(gpu_id, val_loader):
    os.environ['MUSA_VISIBLE_DEVICES'] = str(gpu_id)
    top1_correct = 0
    top5_correct = 0
    total = 0
    log_iter = 100
    total_time = 0.0
    batch_cnt = 0
    # with torch.no_grad():
    for i, (inputs, targets) in enumerate(val_loader):
        # print(targets.size(0))
        if inputs.shape[0] < 24:
            #print("expand", inputs.shape)
            last_image = inputs[-1].unsqueeze(0)
            inputs = torch.cat((inputs, last_image.repeat(8, 1, 1, 1)), dim=0)
            last_tag = targets[-1].unsqueeze(0)
            targets = torch.cat((targets, last_tag.repeat(8)), dim=0)
        # else:
        #     continue
        # torch.set_printoptions(edgeitems=10000, precision=4)
        # print(inputs.shape)
        # print(inputs[0])
        np_dtype = np.float32
        if args.precision == "fp16":
            np_dtype = np.float16
        start_time = time.time()
        outputs = resnet_test.run(['output'], {'input': np.array(inputs, dtype=np_dtype)})[0]
        end_time = time.time()
        total_time += (end_time - start_time)
        outputs = torch.from_numpy(outputs.astype(np.float32))
        
        _, predicted = outputs.topk(5, 1, True, True)
        predicted = predicted.t()
        # print(predicted)
        # print(targets)
        
        total += targets.size(0)
        correct = predicted.eq(targets.view(1, -1).expand_as(predicted))
        
        # print(correct[:1])
        # print(correct[:5])
        
        top1_correct += correct[:1].view(-1).float().sum(0, keepdim=True)
        top5_correct += correct[:5].reshape(-1).float().sum(0, keepdim=True)
        batch_cnt += 1
        if i % log_iter == 0:
            top1_accuracy = 100. * top1_correct / total
            top5_accuracy = 100. * top5_correct / total
            print(f'Top-1 accuracy: {top1_accuracy.item():.2f}%')
            print(f'Top-5 accuracy: {top5_accuracy.item():.2f}%')
        # break

    top1_accuracy = 100. * top1_correct / total
    top5_accuracy = 100. * top5_correct / total
    print('Device: {}, top1 accuracy: {}, batch size is 24, use time: {} Seconds, {} frames per seconds'.format(gpu_id, top1_accuracy.item(), total_time, batch_cnt * 24 / total_time))

    # return top1_accuracy.item(), top5_accuracy.item(), total_time / batch_cnt * 1000.0, batch_cnt * 24 * 1000.0 / total_time

# Perform evaluation
# top1_acc, top5_acc, one_time, throughput = evaluate(infer_dataset)
# print(f'Top-1 accuracy: {top1_acc:.2f}%')
# print(f'Top-5 accuracy: {top5_acc:.2f}%')
# print(f'one batch latency: {one_time:.2f} ms')
# print(f'one batch latency: {throughput:.2f} fps')


def main():
    evaluate(args.gpu_id, infer_dataset)
    # evaluate(0, infer_dataset)
    # gpu_ids = range(1)

    # processes = []
    # for gpu_id in gpu_ids:
    #     p = multiprocessing.Process(target=evaluate, args=(gpu_id, infer_dataset))
    #     processes.append(p)
    #     p.start()

    # # 等待所有进程完成
    # for p in processes:
    #     p.join()

if __name__ == "__main__":
    main()