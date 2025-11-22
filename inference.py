import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import evaluate
import argparse
import yaml

import json

from torch.utils.data import DataLoader, random_split
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch import nn
from tqdm import tqdm
from datasets import disable_caching
from torch.utils.tensorboard import SummaryWriter

from data.oil_spill_dataset import OilSpillDataset


if __name__ == '__main__':

    disable_caching()
    torch.manual_seed(0)
    np.random.seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "config.yaml")
    args = parser.parse_args() 

    with open(args.config) as file:
        cfg = yaml.safe_load(file)

    # Create the current experiment's path
    experiments_path = cfg['EXPERIMENT']['exp_path']
    exp_path = os.path.join(experiments_path, cfg['EXPERIMENT']['name'])

    # Init TensorBoard logger
    tb_logger_path = os.path.join(exp_path, 'tb_logger')
    os.makedirs(tb_logger_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logger_path)

    # Dump the cfg file under the experiment path for better traceability
    cfg_save_path = os.path.join(exp_path, "config.yaml")
    with open(cfg_save_path, 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    # Init the image processor and transform
    image_processor = SegformerImageProcessor(do_rescale=False,
                                              do_normalize=False)
    
    train_dataset = OilSpillDataset(cfg['DATASET_PARAMS']['db_path'], split = 'train', image_processor=image_processor)
    valid_dataset = OilSpillDataset(cfg['DATASET_PARAMS']['db_path'], split = 'test', image_processor=image_processor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['TRAIN_PARAMS']['batch_size'], shuffle=True, num_workers=cfg['TRAIN_PARAMS']['num_workers'])
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)

    # Label and id
    id2label = {"0": "sea", "1": "oil_spill", "2": "look_alike", "3":"ship", "4":"land"}
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # Define the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: ', device)
    ckp_path = cfg['MODEL_PARAMS']['checkpoint']
    print(ckp_path)
    
    if ckp_path and os.path.isfile(ckp_path):
        print(f"Loading model from checkpoint: {ckp_path}")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-" + cfg['MODEL_PARAMS']['model_config'], 
        num_labels=5, id2label=id2label, label2id=label2id)
        model.load_state_dict(torch.load(ckp_path, map_location=device), strict=True)
    else:
        print("No valid checkpoint provided, loading pretrained model.")
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-" + cfg['MODEL_PARAMS']['model_config'], 
                                                                 num_labels=5, 
                                                                 id2label=id2label, 
                                                                 label2id=label2id)
                                                                 
                                                                

    # Move model to device
    model.to(device)       

    os.makedirs(cfg['OUTPUT_PATH']["inference_path"], exist_ok=True)

    # Color palette 
    palette = np.array([[0, 0, 0], # black - sea
                        [0, 255, 255], # cyan - oil spill
                        [255, 0, 0], # red - look-alike
                        [153, 76, 0], # brown - ship
                        [0, 153, 0]])  # green - land
    metric = evaluate.load("mean_iou")
    all_predictions, all_references = [], []
    
    for idx, batch in enumerate(tqdm(valid_dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["original_labels"]
        labels = labels.squeeze().cpu().numpy()

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            # print(outputs.logits)

            upsampled_logits = nn.functional.interpolate(outputs.logits.cpu(), 
                                                         size=labels.shape[-2:], 
                                                         mode="bilinear", 
                                                         align_corners=False)
            predicted = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        
        # Append results
        all_predictions.append(predicted)
        all_references.append(labels)

        predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(650,1250)])[0]
        predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

        h, w = predicted_segmentation_map.shape
        color_seg = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[predicted_segmentation_map == label, :] = color

        # img = np.array(batch["original_image"].squeeze())
        # output_path = os.path.join(cfg['OUTPUT_PATH']["inference_path"], f"output_orig_{idx}.png")
        # plt.imsave(output_path, img.astype(np.uint8))

        # img = np.array(batch["original_image"].squeeze()) * 0.8 + color_seg * 0.2
        img = color_seg
        img = img.astype(np.uint8)

        labels_map = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            labels_map[labels == label, :] = color

        plt.figure(figsize=(5,10))
        plt.subplot(3, 1, 1)
        plt.imshow(np.array(batch["original_image"].squeeze()*255).astype(np.uint8))
        plt.title("Original Image")
        plt.subplot(3, 1, 2)
        plt.imshow(color_seg)
        plt.title("Predicted Segmentation")
        plt.subplot(3, 1, 3)
        plt.imshow(labels_map)
        plt.title("Ground Truth Segmentation")
        # plt.close()
        
        output_path = os.path.join(cfg['OUTPUT_PATH']["inference_path"], f"output_{idx}.png")
        # plt.imsave(output_path, img)
        plt.savefig(output_path)
        plt.close()

    metrics = metric._compute(
                        predictions=np.array(all_predictions),
                        references=np.array(all_references),
                        num_labels=len(id2label),
                        ignore_index=255,
                        reduce_labels=False,
                    )
    
    # Log
    print("Validation Metrics:", metrics)
    metrics_serializable = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in metrics.items()}
    with open(os.path.join(cfg['OUTPUT_PATH']["inference_path"], "metrics.json"), "w") as f:
        json.dump(metrics_serializable, f, indent=4)
    