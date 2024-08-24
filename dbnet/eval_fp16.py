import paddle
import tools.program as program
from ppocr.data import build_dataloader, set_signal_handlers, SimpleDataSet
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
from ppocr.metrics import build_metric
from ppocr.postprocess import build_post_process
import onnxruntime as ort
import numpy as np
import torch
from tqdm import tqdm

import numpy as np
import argparse, time

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-c', type=str, required=True)
parser.add_argument('--gpu_id', '-id', help='Specify gpu id', required=True)
args = parser.parse_args()

def save_array_to_file(array, filename):
    # 展平数组
    flattened_array = array.flatten()
    
    # 保存到文件
    np.savetxt(filename, flattened_array, fmt='%f')

    
    
def main():
    batch_cnt = 0
    sess_options = ort.SessionOptions()

    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # sess_options.optimized_model_filepath = "opt.onnx"
    
    sess = ort.InferenceSession("./model/dbnet-fp16.onnx", sess_options=sess_options, 
                                providers=[('MUSAExecutionProvider', {"prefer_nhwc": '1'})])
    global_config = config["Global"]
    dataset = eval("SimpleDataSet")(config, "Eval", logger, None)
    eval_class = build_metric(config["Metric"])
    post_process_class = build_post_process(config["PostProcess"], global_config)
    batch_sampler = BatchSampler(
            dataset=dataset, batch_size=1, shuffle=False, drop_last=False
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        places=device,
        return_list=True,
    )
    all_batch_numpy = []
    for i, batch in enumerate(data_loader):
        batch_numpy = []
        for item in batch:
            if isinstance(item, paddle.Tensor):
                batch_numpy.append(item.numpy())
            else:
                batch_numpy.append(item)
        # print(batch_numpy[0].dtype)
        all_batch_numpy.append(batch_numpy)
    #  = [[item.numpy() if isinstance(item, paddle.Tensor) else item for item in batch] for batch in dataset]
    # for batch in all_batch_numpy:
    #     for item in batch:
    #         print(item.shape)
    #     exit()
    dataset = [batch[0] for batch in dataset]
    valid_dataloader = DataLoader(
        dataset=dataset,
        batch_size=24, shuffle=False
    )
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    # print(type(valid_dataloader))
    pbar = tqdm(
            total=len(valid_dataloader), desc="eval model:", position=0, leave=True
        )
    #infer 
    # warm up
    for _ in range(2):
        random_input = np.random.randn(24, 3, 736, 1280).astype(np.float16)
        outputs = sess.run(None, {input_name: random_input.astype(np.float16)})[0]
     
    total_time = 0.0
    for idx, images in enumerate(valid_dataloader):
        size_out = images.shape[0]
        images = np.array(images).astype(np.float32)
        #exit()
        if images.shape[0] < 24:
            # print("expand", images.shape)
            additional_array = np.random.rand(4, 3, 736, 1280).astype(np.float32)
            images = np.concatenate((images, additional_array), axis=0)
        batch_cnt += 1
        images = ort.OrtValue.ortvalue_from_numpy(images.astype(np.float16))
        start_time = time.time()
        outputs = sess.run(None, {input_name: images})[0]
        end_time = time.time()
        total_time += (end_time  - start_time)
        outputs = outputs.astype(np.float32)
        # print(outputs.shape)
        for i in range(size_out):
            preds = {
                    "maps": np.expand_dims(outputs[i], axis=0).astype(np.float32)
                }
            batch_numpy = all_batch_numpy[idx * 24 + i]
            post_result = post_process_class(
                preds,
                batch_numpy[1])
            eval_class(post_result, batch_numpy)
        pbar.update(1)
    metric = eval_class.get_metric()       
    # logger.info("metric eval ***************")
    # for k, v in metric.items():
    #     logger.info("{}:{}".format(k, v))
    pbar.close()
    dataset_size = 24 * 20 + 20 # 500
    batch_size = 24
    
    heam_acc = metric['hmean']
    print('Device: {}\ndata type: fp16\ndataset size: {}\nrequired Hmean: 68.00%, Hmean: {:.2f}%\nbatch size is 24\nuse time: {:.2f} Seconds\nlatency: {:.2f}ms/batch\nthroughput: {:.2f} fps'.format(args.gpu_id, dataset_size, heam_acc * 100, total_time, 1000.0 * total_time / batch_cnt, batch_cnt * 24 / total_time))

    # print(heam_acc)

if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
    # print('Device: {}\ndata type: fp16\ndataset size: {}\nrequired top1: 78.00%, top1: {:.2f}%\nbatch size is 24\nuse time: {:.2f} Seconds\nlatency: {:.2f}ms/batch\nthroughput: {:.2f} fps'.format(gpu_id, dataset_size, top1_accuracy.item(), total_time, 1000.0 * total_time / batch_cnt, batch_cnt * 24 / total_time))

