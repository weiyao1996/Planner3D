import itertools
import sys
import numpy as np
import random
import os
import trimesh
import yaml
import json
import csv
import argparse
from PIL import Image
import pickle
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.datasets.threed_front import ThreedFront
from simple_3dviz import Scene
from simple_3dviz.renderables.textured_mesh import TexturedMesh
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render
from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from utils import floor_plan_from_scene, export_scene, get_textured_objects
from graphviz import Digraph

object_ids_csv_path = '/media/ymxlzgy/Data/Dataset/3D-FRONT/object_ids.csv'

def collect_distribution(param7_all):
    means=np.mean(param7_all,axis=0)
    std=np.std(param7_all,axis=0)
    return means, std

def cal_l2_distance(point_1, point_2):
    return np.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

def main(argv):
    parser = argparse.ArgumentParser(
            description="Prepare the 3D-FRONT scene graph"
        )
    parser.add_argument(
            "option",
            help="['bedroom', 'livingroom', 'diningroom', 'library', 'all']"
        )
    parser.add_argument(
        "split",
        help="['train', 'val', 'test', 'all']"
    )
    args = parser.parse_args(argv)
    option = args.option


    path_to_3d_front_dataset_directory="/media/ymxlzgy/Data/Dataset/3D-FRONT/3D-FRONT"
    path_to_3d_future_dataset_directory="/media/ymxlzgy/Data/Dataset/3D-FRONT/3D-FUTURE-model"
    path_to_model_info="/media/ymxlzgy/Data/Dataset/3D-FRONT/3D-FUTURE-model/model_info.json"
    path_to_pickled_3d_futute_models = "/media/ymxlzgy/Data/Dataset/3D-FRONT/3D-FUTURE_pickle/threed_future_model_{}.pkl".format(option)
    path_to_floor_plan_textures = "/media/ymxlzgy/Data/Dataset/3D-FRONT/3D-FRONT-texture"
    d_list = []
    if option == 'all':
        for room_type in ['bedroom', 'livingroom', 'diningroom', 'library']:
            config_file = "/home/ymxlzgy/code/graphto3d_v2/config/{}_sg_config.yaml".format(room_type)
            config = load_config(config_file)
            d = ThreedFront.from_dataset_directory(
                    path_to_3d_front_dataset_directory,
                    path_to_model_info,
                    path_to_3d_future_dataset_directory,
                    path_to_room_masks_dir=None,
                    path_to_bounds=None,
                    filter_fn=filter_function(
                        config["data"],
                        split=config[str(args.split)].get("splits", ["train", "val", "test"])
                    )
                )
            d_list.append(d)
            print("Loaded {0} {1}s".format(len(d.scenes), room_type))
    else:
        config_file = "/home/ymxlzgy/code/graphto3d_v2/config/{}_sg_config.yaml".format(option)
        config = load_config(config_file)
        d_list = [ThreedFront.from_dataset_directory(
            path_to_3d_front_dataset_directory,
            path_to_model_info,
            path_to_3d_future_dataset_directory,
            path_to_room_masks_dir=None,
            path_to_bounds=None,
            filter_fn=filter_function(
                config["data"],
                split=config[str(args.split)].get("splits", ["train", "val", "test"])
            )
        )]
        print("Loaded {0} {1}s".format(len(d.scenes),option))

    obj_box_dict_room = {}
    param7_all_arrays = np.array([]).reshape(-1, 7)
    scene_id_cache = []
    for d in d_list:
        for scene in d.scenes:
            param7_arrays = np.array([]).reshape(-1, 7)
            obj_box_dict = {}
            path_to_objs = os.path.join(
                "/media/ymxlzgy/Data/Dataset/3D-FRONT/visualization",
                "{}".format(scene.scene_id)
            )
            if not os.path.exists(path_to_objs):
                os.mkdir(path_to_objs)
            if scene.scene_id in scene_id_cache:
                continue
            scene_id_cache.append(scene.scene_id)
            # elif os.path.exists(os.path.join(path_to_objs,"obj_boxes.json")):
            #     with open(os.path.join(path_to_objs,"obj_boxes.json")) as file:
            #         obj_box_dict = file.read()
            #         obj_box_dict = json.loads(obj_box_dict)
            #         obj_box_dict_room.update(obj_box_dict)
            #     print("already ", scene.scene_id)
            #     continue
            bbox_param = {}
            for i in range(len(scene.bboxes)):
                size = list(scene.bboxes[i].size * 2) # lhw
                location = list(scene.bboxes[i].position) # xyz, y is up centered at bottom
                angle = [scene.bboxes[i].z_angle]
                param = size + location + angle
                corner_points = scene.bboxes[i].corners().tolist()
                bbox_param[str(i+1)] = {"param7":param, "8points":corner_points, "scale": list(scene.bboxes[i].scale), "model_path": scene.bboxes[i].raw_model_path}
                param7_arrays = np.concatenate((param7_arrays,np.array(param).reshape(-1,7)),axis=0)
            # floor_plan, tr_floor, _ = floor_plan_from_scene(
            #     scene, path_to_floor_plan_textures, without_room_mask=True
            # )
            # f_c = tr_floor[0].centroid
            vertices_f, faces_f = scene.floor_plan
            tr_floor = trimesh.Trimesh(
                np.copy(vertices_f), np.copy(faces_f), process=False
            )
            floor_size = list(tr_floor.extents)
            floor_loc = list(tr_floor.centroid)
            floor_param = floor_size + floor_loc + [0]

            param7_arrays = np.concatenate((param7_arrays, np.array(floor_param).reshape(-1, 7)), axis=0)

            floor_corner_points = trimesh.bounds.corners(tr_floor.bounding_box.bounds).tolist()
            bbox_param[str(len(scene.bboxes)+1)] = {"param7": floor_param, "8points": floor_corner_points, "scale": [1,1,1], "model_path": None}
            param7_arrays[:,3:6] -= scene.centroid
            bbox_param["scene_center"] = scene.centroid.tolist()
            obj_box_dict[scene.scene_id] = bbox_param
            obj_box_dict_room[scene.scene_id] = bbox_param
            with open(os.path.join(path_to_objs, "obj_boxes.json"), "w") as f:
                json.dump(obj_box_dict, f)

            print("generated ", scene.scene_id)
            param7_all_arrays = np.concatenate((param7_all_arrays, param7_arrays), axis=0)
    with open("../GT/3dfront/obj_boxes_{0}_{1}.json".format(option,args.split), "w") as f:
        json.dump(obj_box_dict_room, f)
    mean, std = collect_distribution(param7_all_arrays)

    np.savetxt("../GT/3dfront/boxes_centered_stats_{0}_{1}.txt".format(option,args.split),(mean, std))

if __name__ == "__main__":
    main(sys.argv[1:])