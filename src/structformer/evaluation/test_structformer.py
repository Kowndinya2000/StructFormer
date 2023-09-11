import torch
import numpy as np
import os
import copy
import tqdm
import argparse
from omegaconf import OmegaConf
import time

from torch.utils.data import DataLoader

from structformer.data.tokenizer import Tokenizer
import structformer.data.sequence_dataset as prior_dataset
import structformer.training.train_structformer as prior_model
from structformer.utils.rearrangement import show_pcs
from structformer.evaluation.inference import PointCloudRearrangement
from structformer.evaluation.test_object_selection_network import ObjectSelectionInference


def dist_p2l(p, o, k):
    """(Vectorized meethod) disance, point to line"""
    op = p - o
    k = np.repeat(k, [op.shape[0]]).reshape([2, -1]).T
    op_proj = np.sum(np.multiply(op, k), axis=-1)[..., None] * k
    op_ver = op - op_proj
    return np.linalg.norm(op_ver, axis=-1)

def test_model(model_dir, dirs_cfg):
    prior_inference = PriorInference(model_dir, dirs_cfg, data_split="test")
    prior_inference.validate()


class PriorInference:

    def __init__(self, model_dir, dirs_cfg, data_split="test"):

        cfg, tokenizer, model, optimizer, scheduler, epoch = prior_model.load_model(model_dir, dirs_cfg)

        data_cfg = cfg.dataset

        dataset = prior_dataset.SequenceDataset(data_cfg.dirs, data_cfg.index_dirs, data_split, tokenizer,
                                                data_cfg.max_num_objects,
                                                data_cfg.max_num_other_objects,
                                                data_cfg.max_num_shape_parameters,
                                                data_cfg.max_num_rearrange_features,
                                                data_cfg.max_num_anchor_features,
                                                data_cfg.num_pts,
                                                data_cfg.use_structure_frame)

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        self.cfg = cfg
        self.dataset = dataset
        self.epoch = epoch

    def validate(self):
        """
        validate the pretrained model on the dataset

        :return:
        """
        data_cfg = self.cfg.dataset
        data_iter = DataLoader(self.dataset, batch_size=data_cfg.batch_size, shuffle=False,
                               collate_fn=prior_dataset.SequenceDataset.collate_fn,
                               pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)

        prior_model.validate(self.cfg, self.model, data_iter, self.epoch, self.cfg.device)

    def limited_batch_inference(self, data, verbose=False):
        """
        This function makes the assumption that scenes in the batch have the same number of objects that need to be
        rearranged

        :param data:
        :param model:
        :param test_dataset:
        :param tokenizer:
        :param cfg:
        :param num_samples:
        :param verbose:
        :return:
        """

        data_size = len(data)
        batch_size = self.cfg.dataset.batch_size
        if verbose:
            print("data size:", data_size)
            print("batch size:", batch_size)

        num_batches = int(data_size / batch_size)
        if data_size % batch_size != 0:
            num_batches += 1

        all_obj_preds = []
        all_struct_preds = []
        for b in range(num_batches):
            if b + 1 == num_batches:
                # last batch
                batch = data[b * batch_size:]
            else:
                batch = data[b * batch_size: (b+1) * batch_size]
            data_tensors = [self.dataset.convert_to_tensors(d, self.tokenizer) for d in batch]
            data_tensors = self.dataset.collate_fn(data_tensors)
            predictions = prior_model.infer_once(self.cfg, self.model, data_tensors, self.cfg.device)

            obj_x_preds = torch.cat(predictions["obj_x_outputs"], dim=0)
            obj_y_preds = torch.cat(predictions["obj_y_outputs"], dim=0)
            obj_z_preds = torch.cat(predictions["obj_z_outputs"], dim=0)
            obj_theta_preds = torch.cat(predictions["obj_theta_outputs"], dim=0)
            obj_preds = torch.cat([obj_x_preds, obj_y_preds, obj_z_preds, obj_theta_preds], dim=1)  # batch_size * max num objects, output_dim

            struct_x_preds = torch.cat(predictions["struct_x_inputs"], dim=0)
            struct_y_preds = torch.cat(predictions["struct_y_inputs"], dim=0)
            struct_z_preds = torch.cat(predictions["struct_z_inputs"], dim=0)
            struct_theta_preds = torch.cat(predictions["struct_theta_inputs"], dim=0)
            struct_preds = torch.cat([struct_x_preds, struct_y_preds, struct_z_preds, struct_theta_preds], dim=1) # batch_size, output_dim

            all_obj_preds.append(obj_preds)
            all_struct_preds.append(struct_preds)

        obj_preds = torch.cat(all_obj_preds, dim=0)  # data_size * max num objects, output_dim
        struct_preds = torch.cat(all_struct_preds, dim=0)  # data_size, output_dim

        obj_preds = obj_preds.detach().cpu().numpy()
        struct_preds = struct_preds.detach().cpu().numpy()

        obj_preds = obj_preds.reshape(data_size, -1, obj_preds.shape[-1])  # batch_size, max num objects, output_dim

        return struct_preds, obj_preds


def inference_beam_decoding(model_dir, dirs_cfg, beam_size=100, max_scene_decodes=30000,
                            visualize=True, visualize_action_sequence=False,
                            inference_visualization_dir=None):
    """

    :param model_dir:
    :param beam_size:
    :param max_scene_decodes:
    :param visualize:
    :param visualize_action_sequence:
    :param inference_visualization_dir:
    :param side_view:
    :return:
    """

    if inference_visualization_dir and not os.path.exists(inference_visualization_dir):
        os.makedirs(inference_visualization_dir)

    prior_inference = PriorInference(model_dir, dirs_cfg)
    test_dataset = prior_inference.dataset

    decoded_scene_count = 0
    with tqdm.tqdm(total=len(test_dataset)) as pbar:
        pattern_dists = []
        # for idx in np.random.choice(range(len(test_dataset)), len(test_dataset), replace=False):
        for idx in range(len(test_dataset)):

            if decoded_scene_count == max_scene_decodes:
                break

            filename = test_dataset.get_data_index(idx)
            scene_id = os.path.split(filename)[1][4:-3]

            decoded_scene_count += 1

            ############################################
            # retrieve data
            beam_data = []
            beam_pc_rearrangements = []
            for b in range(beam_size):
                datum = test_dataset.get_raw_data(idx, inference_mode=True, shuffle_object_index=False)

                # not necessary, but just to ensure no test leakage
                datum["struct_x_inputs"] = [0]
                datum["struct_y_inputs"] = [0]
                datum["struct_y_inputs"] = [0]
                datum["struct_theta_inputs"] = [[0] * 9]
                for obj_idx in range(len(datum["obj_x_inputs"])):
                    datum["obj_x_inputs"][obj_idx] = 0
                    datum["obj_y_inputs"][obj_idx] = 0
                    datum["obj_z_inputs"][obj_idx] = 0
                    datum["obj_theta_inputs"][obj_idx] = [0] * 9

                # We can play with different language here
                # datum["sentence"] = modify_language(datum["sentence"], radius=0.5)
                # datum["sentence"] = modify_language(datum["sentence"], position_x=1)
                # datum["sentence"] = modify_language(datum["sentence"], position_y=0.5)

                beam_data.append(datum)
                beam_pc_rearrangements.append(PointCloudRearrangement(datum))

            if visualize:
                datum = beam_data[0]
                # print("#"*50)
                # print("sentence", datum["sentence"])
                # object_selection_inference = ObjectSelectionInference(model_dir, dirs_cfg)
                object_selection_structured_sentence = datum["sentence"][5:]
                structure_specification_structured_sentence = datum["sentence"][:5]
                tokenizer = Tokenizer("/media/exx/T7 Shield/ICLR23/LGMCTS-D/output/struct_diffusion/struct_rearrange/circle/type_vocabs_coarse.json")
                object_selection_natural_sentence = tokenizer.convert_to_natural_sentence(datum["sentence"])
                # structure_specification_natural_sentence =tokenizer.convert_structure_params_to_natural_language(structure_specification_structured_sentence)
                # structure_specification_natural_sentence = object_selection_inference.tokenizer.convert_structure_params_to_natural_language(datum["sentence"][5:])
                # print("natural sentence:", object_selection_natural_sentence)
                # print("datum:", datum)
                # print("structure specification:", structure_specification_natural_sentence)

        

                # show_pcs(datum["xyzs"] + datum["other_xyzs"], datum["rgbs"] + datum["other_rgbs"],
                #          add_coordinate_frame=False, side_view=True, add_table=True)
                # show_pcs(datum["xyzs"] + datum["other_xyzs"], datum["rgbs"] + datum["other_rgbs"],
                #          add_coordinate_frame=True, side_view=True, add_table=True)
                # Commenting out for now
                show_pcs(datum["xyzs"], datum["rgbs"],
                         add_coordinate_frame=True, side_view=False, add_table=True)
                

                show_pcs(datum["xyzs"] + datum["other_xyzs"], datum["rgbs"] + datum["other_rgbs"],
                         add_coordinate_frame=True, side_view=False, add_table=True)
            ############################################
            # autoregressive decoding
            num_target_objects = beam_pc_rearrangements[0].num_target_objects
            # first predict structure pose
            beam_goal_struct_pose, target_object_preds = prior_inference.limited_batch_inference(beam_data)
            for b in range(beam_size):
                datum = beam_data[b]
                datum["struct_x_inputs"] = [beam_goal_struct_pose[b][0]]
                datum["struct_y_inputs"] = [beam_goal_struct_pose[b][1]]
                datum["struct_z_inputs"] = [beam_goal_struct_pose[b][2]]
                datum["struct_theta_inputs"] = [beam_goal_struct_pose[b][3:]]

            # then iteratively predict pose of each object
            beam_goal_obj_poses = []
            for obj_idx in range(num_target_objects):
                struct_preds, target_object_preds = prior_inference.limited_batch_inference(beam_data)
                beam_goal_obj_poses.append(target_object_preds[:, obj_idx])
                for b in range(beam_size):
                    datum = beam_data[b]
                    datum["obj_x_inputs"][obj_idx] = target_object_preds[b][obj_idx][0]
                    datum["obj_y_inputs"][obj_idx] = target_object_preds[b][obj_idx][1]
                    datum["obj_z_inputs"][obj_idx] = target_object_preds[b][obj_idx][2]
                    datum["obj_theta_inputs"][obj_idx] = target_object_preds[b][obj_idx][3:]
            # concat in the object dim
            beam_goal_obj_poses = np.stack(beam_goal_obj_poses, axis=0)
            # swap axis
            beam_goal_obj_poses = np.swapaxes(beam_goal_obj_poses, 1, 0)  # batch size, number of target objects, pose dim

            ############################################
            # move pc
            for bi in range(beam_size):
                beam_pc_rearrangements[bi].set_goal_poses(beam_goal_struct_pose[bi], beam_goal_obj_poses[bi])
                beam_pc_rearrangements[bi].rearrange() 
                obj_poses_pattern = []
                for gpose in beam_pc_rearrangements[bi].goal_poses["obj_poses"]:
                    # gpose = gpose.detach().cpu().numpy()
                    goal_pose = np.eye(4)
                    goal_pose[:3, 3] = gpose[:3]
                    print("translation:", gpose[:3])
                    goal_pose[:3, :3] = np.array(gpose[3:]).reshape(3, 3)   
                    assert np.allclose(np.linalg.det(goal_pose[:3, :3]), 1.0)
                    obj_poses_pattern.append(goal_pose)                                                
                print("---------------------------------------\n")
                obj_poses_pattern = np.vstack(obj_poses_pattern)
                # get the up most and low most points first"""
                # lo_idx = np.argmax(obj_poses_pattern[:, 1], axis=-1)
                # hi_idx = np.argmin(obj_poses_pattern[:, 1], axis=-1)
                # lo_pose = obj_poses_pattern[lo_idx, :2]
                # hi_pose = obj_poses_pattern[hi_idx, :2]

                lo_idx = np.argmax(obj_poses_pattern[:, 0], axis=0)
                hi_idx = np.argmin(obj_poses_pattern[:, 0], axis=0)
                lo_pose = obj_poses_pattern[lo_idx, [0, 2]]
                hi_pose = obj_poses_pattern[hi_idx, [0, 2]]
                
                k = (hi_pose - lo_pose) / np.linalg.norm(hi_pose - lo_pose)
                o = hi_pose
                threshold = 0.1 
                # dists = dist_p2l(obj_poses_pattern[:, :2], o[None, :], k[None, :])
                dists = dist_p2l(obj_poses_pattern[:, [0, 2]], o[None, :], k[None, :])
                pattern_dists.append(np.max(dists))
                status = not (np.max(dists) > threshold)
                if not status:
                    print("Max Dist:", np.max(dists), "status:", status)
                    pass
                
                # print("goal pose:", gpose)
            ############################################
            if visualize:
                for pc_rearrangement in beam_pc_rearrangements:
                    # print("I am here at 250")
                    # Commenting out for now
                    pc_rearrangement.visualize("goal", add_other_objects=False,
                                               add_coordinate_frame=True, side_view=False, add_table=False)
                    pc_rearrangement.visualize("goal", add_other_objects=True,
                                               add_coordinate_frame=True, side_view=False, add_table=False)
                    pass
            if inference_visualization_dir:
                for pc_rearrangement in beam_pc_rearrangements:
                    # print("I am here at 255")
                    pc_rearrangement.visualize("goal", add_other_objects=True,
                                               add_coordinate_frame=False, side_view=False, add_table=False,
                                               save_vis=True,
                                               save_filename=os.path.join(inference_visualization_dir, "{}.jpg".format(scene_id)))

            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--dataset_base_dir", help='location of the dataset', type=str)
    parser.add_argument("--model_dir", help='location for the saved model', type=str)
    parser.add_argument("--dirs_config", help='config yaml file for directories', default="", type=str)
    args = parser.parse_args()

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")

    # # debug only
    # args.dataset_base_dir = "/home/weiyu/data_drive/data_new_objects_test_split"
    # args.model_dir = "/home/weiyu/Research/intern/StructFormer/models/structformer_line/best_model"
    # args.dirs_config = "/home/weiyu/Research/intern/StructFormer/structformer/configs/data/line_dirs.yaml"

    if args.dirs_config:
        assert os.path.exists(args.dirs_config), "Cannot find config yaml file at {}".format(args.dirs_config)
        dirs_cfg = OmegaConf.load(args.dirs_config)
        dirs_cfg.dataset_base_dir = args.dataset_base_dir
        OmegaConf.resolve(dirs_cfg)
    else:
        dirs_cfg = None

    inference_beam_decoding(args.model_dir, dirs_cfg, beam_size=3, max_scene_decodes=30000,
                            visualize=True, visualize_action_sequence=False,
                            inference_visualization_dir=None)
