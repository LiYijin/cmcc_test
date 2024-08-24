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

def main():
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
    sess = ort.InferenceSession("./infer-mv3-db/model.onnx", providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(type(valid_dataloader))
    pbar = tqdm(
            total=len(valid_dataloader), desc="eval model:", position=0, leave=True
        )
    #infer 
    for idx, images in enumerate(valid_dataloader):
        # print(idx, images.shape)
        outputs = sess.run(None, {input_name: np.array(images)})[0]
        # print(outputs.shape)
        for i in range(outputs.shape[0]):
            preds = {
                    "maps": np.expand_dims(outputs[i], axis=0).astype(np.float32)
                }
            batch_numpy = all_batch_numpy[idx * 24 + i]
            # print(type(preds))
            # print(type(batch_numpy[1]))
            # print(preds, batch_numpy[1].shape)
            # exit()
            post_result = post_process_class(
                preds,
                batch_numpy[1])
            eval_class(post_result, batch_numpy)
        pbar.update(1)
    metric = eval_class.get_metric()       
    logger.info("metric eval ***************")
    for k, v in metric.items():
        logger.info("{}:{}".format(k, v))

if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()