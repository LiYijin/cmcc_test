import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import onnxruntime as ort
import onnxruntime.capi as ort_cap
import numpy as np
import random
import time
import os
# Define transform for the input images
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(288),
    transforms.ToTensor(),
    normalize,
])

# Load the ImageNet validation dataset
val_dataset = datasets.ImageFolder(
    '/dataset/dataset-2012/val',
    transform=val_transform
)

# Create a DataLoader for the validation dataset
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

resnet_test = ort.InferenceSession("./efficientnetv2_rw_t_fp16_24.onnx", providers=['MUSAExecutionProvider'])
#options.prefer_nhwc = 1;
#options.hwcn_conv_weight = 1;

# Evaluation function
def evaluate(gpu_id, val_loader):
    os.environ['MUSA_VISIBLE_DEVICES'] = str(gpu_id)
    top1_correct = 0
    top5_correct = 0
    total = 0
    log_iter = 1000
    total_time = 0.0
    batch_cnt = 0
    for i, (inputs, targets) in enumerate(val_loader):
        if inputs.shape[0] < 24:
            last_image = inputs[-1].unsqueeze(0)
            inputs = torch.cat((inputs, last_image.repeat(16, 1, 1, 1)), dim=0)
            last_tag = targets[-1].unsqueeze(0)
            targets = torch.cat((targets, last_tag.repeat(16)), dim=0)
        start_time = time.time()
        outputs = resnet_test.run(['1354'], {'input.1': np.array(inputs, dtype=np.float16)})[0]
        end_time = time.time()
        total_time += (end_time - start_time)
        outputs = outputs.astype(float)
        outputs = torch.from_numpy(outputs)
        _, predicted = outputs.topk(5, 1, True, True)
        predicted = predicted.t()

        total += targets.size(0)
        correct = predicted.eq(targets.view(1, -1).expand_as(predicted))

        top1_correct += correct[:1].reshape(-1).float().sum(0, keepdim=True)
        top5_correct += correct[:5].reshape(-1).float().sum(0, keepdim=True)
        top1_accuracy = 100. * top1_correct / total
        top5_accuracy = 100. * top5_correct / total
        batch_cnt += 1
        print(f'Top-1 accuracy: {top1_accuracy.item():.2f}%')
        print(f'Top-5 accuracy: {top5_accuracy.item():.2f}%')
        print("average time: ", total_time / batch_cnt * 1000.0, "ms") 
    top1_accuracy = 100. * top1_correct / total
    top5_accuracy = 100. * top5_correct / total
    print('Device: {}, Top-1 accuracy {}, batch size is 24, use time: {} Seconds, {} frames per seconds'.format(gpu_id, top1_accuracy.item(), total_time, batch_cnt * 24 *1000.0 / total_time))

    # return top1_accuracy.item(), top5_accuracy.item(), total_time / batch_cnt * 1000.0, batch_cnt * 24 * 1000.0 / total_time

# # Perform evaluation
# top1_acc, top5_acc, one_time, throughput = evaluate(val_loader)
# print(f'Top-1 accuracy: {top1_acc:.2f}%')
# print(f'Top-5 accuracy: {top5_acc:.2f}%')
# print(f'one batch latency: {one_time:.2f} ms')
# print(f'one batch latency: {throughput:.2f} fps')

def main():
    evaluate(0, val_loader)
    # gpu_ids = range(1)

    # processes = []
    # for gpu_id in gpu_ids:
    #     p = multiprocessing.Process(target=evaluate, args=(gpu_id, val_loader))
    #     processes.append(p)
    #     p.start()

    # # 等待所有进程完成
    # for p in processes:
    #     p.join()

if __name__ == "__main__":
    main()
