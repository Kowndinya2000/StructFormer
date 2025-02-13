from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import tqdm

import pickle
import argparse
from omegaconf import OmegaConf
from collections import defaultdict

from torch.utils.data import DataLoader
from structformer.data.sequence_dataset import SequenceDataset
from structformer.models.pose_generation_network import PriorContinuousOutEncoderDecoderStructPCT6DDropoutAllObjects
from structformer.data.tokenizer import Tokenizer
from structformer.utils.rearrangement import evaluate_prior_prediction, generate_square_subsequent_mask

def dist_p2l(p, o, k):
    """(Vectorized meethod) disance, point to line"""
    op = p - o
    k = np.repeat(k, [op.shape[0]]).reshape([2, -1]).T
    op_proj = np.sum(np.multiply(op, k), axis=-1)[..., None] * k
    op_ver = op - op_proj
    return np.linalg.norm(op_ver, axis=-1)
def train_model(cfg, model, data_iter, optimizer, warmup, num_epochs, device, save_best_model, grad_clipping=1.0):

    if save_best_model:
        best_model_dir = os.path.join(cfg.experiment_dir, "best_model")
        print("best model will be saved to {}".format(best_model_dir))
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        best_score = -np.inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        epoch_loss = 0
        gts = defaultdict(list)
        predictions = defaultdict(list)

        with tqdm.tqdm(total=len(data_iter["train"])) as pbar:
            pattern_dists = []
            for step, batch in enumerate(data_iter["train"]):
                optimizer.zero_grad()
                # input
                xyzs = batch["xyzs"].to(device, non_blocking=True)
                rgbs = batch["rgbs"].to(device, non_blocking=True)
                object_pad_mask = batch["object_pad_mask"].to(device, non_blocking=True)
                other_xyzs = batch["other_xyzs"].to(device, non_blocking=True)
                other_rgbs = batch["other_rgbs"].to(device, non_blocking=True)
                other_object_pad_mask = batch["other_object_pad_mask"].to(device, non_blocking=True)
                sentence = batch["sentence"].to(device, non_blocking=True)
                sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)
                token_type_index = batch["token_type_index"].to(device, non_blocking=True)
                position_index = batch["position_index"].to(device, non_blocking=True)
                obj_x_inputs = batch["obj_x_inputs"].to(device, non_blocking=True)
                obj_y_inputs = batch["obj_y_inputs"].to(device, non_blocking=True)
                obj_z_inputs = batch["obj_z_inputs"].to(device, non_blocking=True)
                obj_theta_inputs = batch["obj_theta_inputs"].to(device, non_blocking=True)

                struct_x_inputs = batch["struct_x_inputs"].to(device, non_blocking=True)
                struct_y_inputs = batch["struct_y_inputs"].to(device, non_blocking=True)
                struct_z_inputs = batch["struct_z_inputs"].to(device, non_blocking=True)
                struct_theta_inputs = batch["struct_theta_inputs"].to(device, non_blocking=True)
                struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
                struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
                struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)

                tgt_mask = generate_square_subsequent_mask(object_pad_mask.shape[1] + 1).to(device, non_blocking=True)
                start_token = torch.zeros((object_pad_mask.shape[0], 1), dtype=torch.long).to(device, non_blocking=True)

                # output
                targets = {}
                for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                            "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
                    targets[key] = batch[key].to(device, non_blocking=True)
                    targets[key] = targets[key].reshape(targets[key].shape[0] * targets[key].shape[1], -1)

                preds = model.forward(xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                                      sentence, sentence_pad_mask, token_type_index,
                                      obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index,
                                      tgt_mask, start_token,
                                      struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                                      struct_position_index, struct_token_type_index, struct_pad_mask)
                # print("Predict X Shape:", preds["obj_x_outputs"].shape)
                # print("Predict Y Shape:", preds["obj_y_outputs"].shape)
                # print("Predict Z Shape:", preds["obj_z_outputs"].shape)
                # print("Predict Theta Shape:", preds["obj_theta_outputs"].shape)

                # print("Prediction X:", preds["obj_x_outputs"][0])
                # print("Prediction Y:", preds["obj_y_outputs"][0])
                # print("Prediction Z:", preds["obj_z_outputs"][0])
                # print("Prediction Theta:", preds["obj_theta_outputs"][0])

                for num in range(cfg.dataset.batch_size):
                    obj_poses_pattern = []
                    for ind in range(0, cfg.dataset.max_num_objects):
                        goal_pose = np.eye(4)
                        goal_pose = goal_pose[:3]
                        goal_pose[0, 3] = preds["obj_x_outputs"][ind]
                        goal_pose[1, 3] = preds["obj_y_outputs"][ind]
                        goal_pose[2, 3] = preds["obj_z_outputs"][ind]
                        goal_pose[:3, :3] = preds["obj_theta_outputs"][ind].reshape(3, 3).detach().numpy() 
                        # rotation_offset = np.transpose(obj_theta_inputs[num, ind].reshape(3, 3).detach().numpy()) @ preds["obj_theta_outputs"][ind].reshape(3, 3).detach().numpy()
                        trans_offset = goal_pose[:3, 3] - np.array([obj_x_inputs[num, ind].detach().numpy(), obj_y_inputs[num, ind].detach().numpy(), obj_z_inputs[num, ind].detach().numpy()])
                        # trans_offset = np.array([0., 0., 0.])
                        # offset = np.eye(4)
                        # offset[:3, :3] = rotation_offset
                        # offset[:3, 3] = trans_offset
                        # rot_new = goal_pose[:3, :3] @ rotation_offset
                        # trans_new = goal_pose[:3, :3] @ trans_offset + goal_pose[:3, 3]
                        # goal_pose[:3, :3] = rot_new 
                        # goal_pose[:3, 3] = trans_new 
                        goal_pose[:3, 3] = trans_offset
                        # + obj_theta_inputs[num, ind].reshape(3, 3).detach().numpy()
                        assert np.allclose(np.linalg.det(goal_pose[:3, :3]), 1.0)
                        obj_poses_pattern.append(goal_pose)
                                                
                    obj_poses_pattern = np.vstack(obj_poses_pattern)
                    # get the up most and low most points first"""
                    lo_idx = np.argmax(obj_poses_pattern[:, 1], axis=-1)
                    hi_idx = np.argmin(obj_poses_pattern[:, 1], axis=-1)
                    lo_pose = obj_poses_pattern[lo_idx, :2]
                    hi_pose = obj_poses_pattern[hi_idx, :2]
                    k = (hi_pose - lo_pose) / np.linalg.norm(hi_pose - lo_pose)
                    o = hi_pose
                    threshold = 0.1 
                    dists = dist_p2l(obj_poses_pattern[:, :2], o[None, :], k[None, :])
                    pattern_dists.append(np.max(dists))
                    status = not (np.max(dists) > threshold)
                    if not status:
                        # print("Line pattern check failed!")
                        pass
                    # print("status:", status)
                

                loss = model.criterion(preds, targets)
                loss.backward()

                if grad_clipping != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

                optimizer.step()
                epoch_loss += loss

                for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                            "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
                    gts[key].append(targets[key].detach())
                    predictions[key].append(preds[key].detach())

                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss})

        warmup.step()

        print('[Epoch:{}]:  Training Loss:{:.4}'.format(epoch, epoch_loss))
        evaluate_prior_prediction(gts, predictions, ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                                    "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"])

        score = validate(cfg, model, data_iter["valid"], epoch, device)
        if save_best_model and score > best_score:
            print("Saving best model so far...")
            best_score = score
            save_model(best_model_dir, cfg, epoch, model)
    print("Max dist:", max(pattern_dists))
    print("Min dist:", min(pattern_dists))
    return model    


def validate(cfg, model, data_iter, epoch, device):
    """
    helper function to evaluate the model

    :param model:
    :param data_iter:
    :param epoch:
    :param device:
    :return:
    """

    model.eval()

    epoch_loss = 0
    gts = defaultdict(list)
    predictions = defaultdict(list)
    with torch.no_grad():

        with tqdm.tqdm(total=len(data_iter)) as pbar:
            for step, batch in enumerate(data_iter):

                # input
                xyzs = batch["xyzs"].to(device, non_blocking=True)
                rgbs = batch["rgbs"].to(device, non_blocking=True)
                object_pad_mask = batch["object_pad_mask"].to(device, non_blocking=True)
                other_xyzs = batch["other_xyzs"].to(device, non_blocking=True)
                other_rgbs = batch["other_rgbs"].to(device, non_blocking=True)
                other_object_pad_mask = batch["other_object_pad_mask"].to(device, non_blocking=True)
                sentence = batch["sentence"].to(device, non_blocking=True)
                sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)
                token_type_index = batch["token_type_index"].to(device, non_blocking=True)
                position_index = batch["position_index"].to(device, non_blocking=True)
                obj_x_inputs = batch["obj_x_inputs"].to(device, non_blocking=True)
                obj_y_inputs = batch["obj_y_inputs"].to(device, non_blocking=True)
                obj_z_inputs = batch["obj_z_inputs"].to(device, non_blocking=True)
                obj_theta_inputs = batch["obj_theta_inputs"].to(device, non_blocking=True)

                struct_x_inputs = batch["struct_x_inputs"].to(device, non_blocking=True)
                struct_y_inputs = batch["struct_y_inputs"].to(device, non_blocking=True)
                struct_z_inputs = batch["struct_z_inputs"].to(device, non_blocking=True)
                struct_theta_inputs = batch["struct_theta_inputs"].to(device, non_blocking=True)
                struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
                struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
                struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)

                tgt_mask = generate_square_subsequent_mask(object_pad_mask.shape[1] + 1).to(device, non_blocking=True)
                start_token = torch.zeros((object_pad_mask.shape[0], 1), dtype=torch.long).to(device, non_blocking=True)

                # output
                targets = {}
                for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                            "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
                    targets[key] = batch[key].to(device, non_blocking=True)
                    targets[key] = targets[key].reshape(targets[key].shape[0] * targets[key].shape[1], -1)

                preds = model.forward(xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                                      sentence, sentence_pad_mask, token_type_index,
                                      obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index,
                                      tgt_mask, start_token,
                                      struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                                      struct_position_index, struct_token_type_index, struct_pad_mask)
                loss = model.criterion(preds, targets)

                for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                            "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
                    gts[key].append(targets[key])
                    predictions[key].append(preds[key])

                epoch_loss += loss
                pbar.update(1)
                pbar.set_postfix({"Batch loss": loss})

    print('[Epoch:{}]:  Val Loss:{:.4}'.format(epoch, epoch_loss))

    score = evaluate_prior_prediction(gts, predictions,
                     ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                      "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"])
    return score


def infer_once(cfg, model, batch, device):

    model.eval()

    predictions = defaultdict(list)
    with torch.no_grad():

        # input
        xyzs = batch["xyzs"].to(device, non_blocking=True)
        rgbs = batch["rgbs"].to(device, non_blocking=True)
        object_pad_mask = batch["object_pad_mask"].to(device, non_blocking=True)
        other_xyzs = batch["other_xyzs"].to(device, non_blocking=True)
        other_rgbs = batch["other_rgbs"].to(device, non_blocking=True)
        other_object_pad_mask = batch["other_object_pad_mask"].to(device, non_blocking=True)
        sentence = batch["sentence"].to(device, non_blocking=True)
        sentence_pad_mask = batch["sentence_pad_mask"].to(device, non_blocking=True)
        token_type_index = batch["token_type_index"].to(device, non_blocking=True)
        position_index = batch["position_index"].to(device, non_blocking=True)

        struct_position_index = batch["struct_position_index"].to(device, non_blocking=True)
        struct_token_type_index = batch["struct_token_type_index"].to(device, non_blocking=True)
        struct_pad_mask = batch["struct_pad_mask"].to(device, non_blocking=True)

        obj_x_inputs = batch["obj_x_inputs"].to(device, non_blocking=True)
        obj_y_inputs = batch["obj_y_inputs"].to(device, non_blocking=True)
        obj_z_inputs = batch["obj_z_inputs"].to(device, non_blocking=True)
        obj_theta_inputs = batch["obj_theta_inputs"].to(device, non_blocking=True)

        struct_x_inputs = batch["struct_x_inputs"].to(device, non_blocking=True)
        struct_y_inputs = batch["struct_y_inputs"].to(device, non_blocking=True)
        struct_z_inputs = batch["struct_z_inputs"].to(device, non_blocking=True)
        struct_theta_inputs = batch["struct_theta_inputs"].to(device, non_blocking=True)

        tgt_mask = generate_square_subsequent_mask(object_pad_mask.shape[1] + 1).to(device, non_blocking=True)
        start_token = torch.zeros((object_pad_mask.shape[0], 1), dtype=torch.long).to(device, non_blocking=True)

        preds = model.forward(xyzs, rgbs, object_pad_mask, other_xyzs, other_rgbs, other_object_pad_mask,
                              sentence, sentence_pad_mask, token_type_index,
                              obj_x_inputs, obj_y_inputs, obj_z_inputs, obj_theta_inputs, position_index,
                              tgt_mask, start_token,
                              struct_x_inputs, struct_y_inputs, struct_z_inputs, struct_theta_inputs,
                              struct_position_index, struct_token_type_index, struct_pad_mask)

        for key in ["obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs",
                    "struct_x_inputs", "struct_y_inputs", "struct_z_inputs", "struct_theta_inputs"]:
            predictions[key].append(preds[key])

    return predictions


def save_model(model_dir, cfg, epoch, model, optimizer=None, scheduler=None):
    state_dict = {'epoch': epoch,
                  'model_state_dict': model.state_dict()}
    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state_dict, os.path.join(model_dir, "model.tar"))
    OmegaConf.save(cfg, os.path.join(model_dir, "config.yaml"))


def load_model(model_dir, dirs_cfg):
    """
    Load transformer model
    Important: to use the model, call model.eval() or model.train()
    :param model_dir:
    :return:
    """
    # load dictionaries
    cfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    if dirs_cfg:
        cfg = OmegaConf.merge(cfg, dirs_cfg)

    data_cfg = cfg.dataset
    tokenizer = Tokenizer(data_cfg.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    # initialize model
    model_cfg = cfg.model
    model = PriorContinuousOutEncoderDecoderStructPCT6DDropoutAllObjects(vocab_size,
                                                                         num_attention_heads=model_cfg.num_attention_heads,
                                                                         encoder_hidden_dim=model_cfg.encoder_hidden_dim,
                                                                         encoder_dropout=model_cfg.encoder_dropout,
                                                                         encoder_activation=model_cfg.encoder_activation,
                                                                         encoder_num_layers=model_cfg.encoder_num_layers,
                                                                         structure_dropout=model_cfg.structure_dropout,
                                                                         object_dropout=model_cfg.object_dropout,
                                                                         theta_loss_divide=model_cfg.theta_loss_divide,
                                                                         ignore_rgb=model_cfg.ignore_rgb)
    model.to(cfg.device)

    # load state dicts
    checkpoint = torch.load(os.path.join(model_dir, "model.tar"))
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = None
    if "optimizer_state_dict" in checkpoint:
        training_cfg = cfg.training
        optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = None
    if "scheduler_state_dict" in checkpoint:
        scheduler = None
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    return cfg, tokenizer, model, optimizer, scheduler, epoch


def run_model(cfg):

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)
        torch.backends.cudnn.deterministic = True

    data_cfg = cfg.dataset
    tokenizer = Tokenizer(data_cfg.vocab_dir)
    vocab_size = tokenizer.get_vocab_size()

    train_dataset = SequenceDataset(data_cfg.dirs, data_cfg.index_dirs, "train", tokenizer,
                                    data_cfg.max_num_objects,
                                    data_cfg.max_num_other_objects,
                                    data_cfg.max_num_shape_parameters,
                                    data_cfg.max_num_rearrange_features,
                                    data_cfg.max_num_anchor_features,
                                    data_cfg.num_pts,
                                    data_cfg.use_structure_frame)
    valid_dataset = SequenceDataset(data_cfg.dirs, data_cfg.index_dirs, "valid", tokenizer,
                                    data_cfg.max_num_objects,
                                    data_cfg.max_num_other_objects,
                                    data_cfg.max_num_shape_parameters,
                                    data_cfg.max_num_rearrange_features,
                                    data_cfg.max_num_anchor_features,
                                    data_cfg.num_pts,
                                    data_cfg.use_structure_frame)

    data_iter = {}
    data_iter["train"] = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True,
                                    collate_fn=SequenceDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)
    data_iter["valid"] = DataLoader(valid_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                    collate_fn=SequenceDataset.collate_fn,
                                    pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)

    # load model
    model_cfg = cfg.model
    model = PriorContinuousOutEncoderDecoderStructPCT6DDropoutAllObjects(vocab_size,
                                                                         num_attention_heads=model_cfg.num_attention_heads,
                                                                         encoder_hidden_dim=model_cfg.encoder_hidden_dim,
                                                                         encoder_dropout=model_cfg.encoder_dropout,
                                                                         encoder_activation=model_cfg.encoder_activation,
                                                                         encoder_num_layers=model_cfg.encoder_num_layers,
                                                                         structure_dropout=model_cfg.structure_dropout,
                                                                         object_dropout=model_cfg.object_dropout,
                                                                         theta_loss_divide=model_cfg.theta_loss_divide,
                                                                         ignore_rgb=model_cfg.ignore_rgb)
    model.to(cfg.device)

    training_cfg = cfg.training
    optimizer = optim.Adam(model.parameters(), lr=training_cfg.learning_rate, weight_decay=training_cfg.l2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg.lr_restart)
    warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=training_cfg.warmup,
                                    after_scheduler=scheduler)

    train_model(cfg, model, data_iter, optimizer, warmup, training_cfg.max_epochs, cfg.device, cfg.save_best_model)

    # save model
    if cfg.save_model:
        model_dir = os.path.join(cfg.experiment_dir, "model")
        print("Saving model to {}".format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_model(model_dir, cfg, cfg.training.max_epochs, model, optimizer, scheduler)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--dataset_base_dir", help='location of the dataset', type=str)
    parser.add_argument("--main_config", help='config yaml file for the model',
                        default='../configs/structformer.yaml',
                        type=str)
    parser.add_argument("--dirs_config", help='config yaml file for directories',
                        default='../configs/data/circle_dirs.yaml',
                        type=str)
    args = parser.parse_args()

    # # debug
    # args.dataset_base_dir = "/home/weiyu/data_drive/data_new_objects"

    assert os.path.exists(args.main_config), "Cannot find config yaml file at {}".format(args.main_config)
    assert os.path.exists(args.dirs_config), "Cannot find config yaml file at {}".format(args.dir_config)

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")

    main_cfg = OmegaConf.load(args.main_config)
    dirs_cfg = OmegaConf.load(args.dirs_config)
    cfg = OmegaConf.merge(main_cfg, dirs_cfg)
    cfg.dataset_base_dir = args.dataset_base_dir
    OmegaConf.resolve(cfg)

    if not os.path.exists(cfg.experiment_dir):
        os.makedirs(cfg.experiment_dir)

    OmegaConf.save(cfg, os.path.join(cfg.experiment_dir, "config.yaml"))

    run_model(cfg)