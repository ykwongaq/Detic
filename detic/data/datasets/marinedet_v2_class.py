# Copyright (c) Facebook, Inc. and its affiliates.
# import logging
# import os
#
# from detectron2.data.datasets.register_coco import register_coco_instances
#
#
# _CUSTOM_SPLITS_COCO = {
#     "marinedet_v2_class_level_train": ("marinedet_v2/images", "marinedet_v2/annotations/class_level_train.json"),
#     "marinedet_v2_class_level_val": ("marinedet_v2/images", "marinedet_v2/annotations/class_level_val.json"),
# }
#
# for key, (image_root, json_file) in _CUSTOM_SPLITS_COCO.items():
#     # Assume pre-defined datasets live in `./datasets`.
#     register_coco_instances(
#         key,
#         {}, # empty metadata, it will be overwritten in load_coco_json() function
#         os.path.join("datasets", json_file) if "://" not in json_file else json_file,
#         os.path.join("datasets", image_root),
#     )

# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os

from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .lvis_v1 import custom_register_lvis_instances


categories_seen = [
{"id": 0, "name": "Teleostei"},
{"id": 1, "name": "Malacostraca"},
{"id": 2, "name": "Elasmobranchii"},
{"id": 3, "name": "Gastropoda"},
{"id": 6, "name": "Actinopterygii"},
{"id": 8, "name": "Bivalvia"},
{"id": 9, "name": "Reptilia"},
{"id": 10, "name": "Hexacorallia"},
{"id": 11, "name": "Aves"},
{"id": 13, "name": "Echinoidea"},
{"id": 14, "name": "Octocorallia"},
{"id": 15, "name": "Polychaeta"},
{"id": 16, "name": "Hydrozoa"},
{"id": 17, "name": "Chondrichthyes"},
{"id": 18, "name": "Demospongiae"},
{"id": 21, "name": "Dipneusti"},
{"id": 22, "name": "Cubozoa"},
{"id": 23, "name": "Hexapoda"},
{"id": 24, "name": "Ophiuroidea"},
{"id": 26, "name": "Coelacanthi"},
{"id": 27, "name": "Nuda"},
{"id": 29, "name": "Petromyzonti"},
{"id": 30, "name": "Ascidiacea"},
{"id": 32, "name": "Florideophyceae"},
]

categories_unseen = [
    {"id": 0, "name": "Teleostei"},
    {"id": 1, "name": "Malacostraca"},
    {"id": 2, "name": "Elasmobranchii"},
    {"id": 3, "name": "Gastropoda"},
    {"id": 6, "name": "Actinopterygii"},
    {"id": 8, "name": "Bivalvia"},
    {"id": 9, "name": "Reptilia"},
    {"id": 10, "name": "Hexacorallia"},
    {"id": 11, "name": "Aves"},
    {"id": 13, "name": "Echinoidea"},
    {"id": 14, "name": "Octocorallia"},
    {"id": 15, "name": "Polychaeta"},
    {"id": 16, "name": "Hydrozoa"},
    {"id": 17, "name": "Chondrichthyes"},
    {"id": 18, "name": "Demospongiae"},
    {"id": 21, "name": "Dipneusti"},
    {"id": 22, "name": "Cubozoa"},
    {"id": 23, "name": "Hexapoda"},
    {"id": 24, "name": "Ophiuroidea"},
    {"id": 26, "name": "Coelacanthi"},
    {"id": 27, "name": "Nuda"},
    {"id": 29, "name": "Petromyzonti"},
    {"id": 30, "name": "Ascidiacea"},
    {"id": 32, "name": "Florideophyceae"},
    {"id": 4, "name": "Mammalia"},
    {"id": 5, "name": "Cephalopoda"},
    {"id": 7, "name": "Scyphozoa"},
    {"id": 12, "name": "Holothuroidea"},
    {"id": 19, "name": "Asteroidea"},
    {"id": 20, "name": "Crinoidea"},
    {"id": 25, "name": "Chondrostei"},
    {"id": 28, "name": "Myxini"},
    {"id": 31, "name": "Pycnogonida"},
]

category_image_count_train = [
    {"id": 0, "image_count": 12894},
{"id": 1, "image_count": 2160},
{"id": 2, "image_count": 1868},
{"id": 3, "image_count": 1407},
{"id": 6, "image_count": 641},
{"id": 8, "image_count": 1681},
{"id": 9, "image_count": 338},
{"id": 10, "image_count": 735},
{"id": 11, "image_count": 791},
{"id": 13, "image_count": 539},
{"id": 14, "image_count": 3173},
{"id": 15, "image_count": 305},
{"id": 16, "image_count": 469},
{"id": 17, "image_count": 60},
{"id": 18, "image_count": 162},
{"id": 21, "image_count": 109},
{"id": 22, "image_count": 56},
{"id": 23, "image_count": 22},
{"id": 24, "image_count": 44},
{"id": 26, "image_count": 36},
{"id": 27, "image_count": 24},
{"id": 29, "image_count": 41},
{"id": 30, "image_count": 2},
{"id": 32, "image_count": 3},
]

category_image_count_val = [
    {"id": 16, "image_count": 83},
    {"id": 2, "image_count": 405},
    {"id": 0, "image_count": 4192},
    {"id": 14, "image_count": 1440},
    {"id": 32, "image_count": 23},
    {"id": 9, "image_count": 86},
    {"id": 1, "image_count": 539},
    {"id": 15, "image_count": 111},
    {"id": 18, "image_count": 77},
    {"id": 13, "image_count": 384},
    {"id": 10, "image_count": 245},
    {"id": 8, "image_count": 458},
    {"id": 3, "image_count": 343},
    {"id": 24, "image_count": 17},
    {"id": 11, "image_count": 441},
    {"id": 6, "image_count": 122},
    {"id": 21, "image_count": 40},
    {"id": 29, "image_count": 7},
    {"id": 22, "image_count": 13},
    {"id": 23, "image_count": 12},
    {"id": 26, "image_count": 4},
    {"id": 27, "image_count": 2},
    {"id": 17, "image_count": 27},
    {"id": 5, "image_count": 1109},
    {"id": 4, "image_count": 1971},
    {"id": 19, "image_count": 378},
    {"id": 7, "image_count": 1411},
    {"id": 25, "image_count": 129},
    {"id": 12, "image_count": 378},
    {"id": 20, "image_count": 71},
    {"id": 31, "image_count": 2},
    {"id": 28, "image_count": 20},
]
# MARINEDET_V2_CATEGORY_IMAGE_COUNT = [
#      {"id": 0, "image_count": 4237},
#      {"id": 1, "image_count": 1380},
#      {"id": 2, "image_count": 1006},
#      {"id": 3, "image_count": 774},
#      {"id": 6, "image_count": 346},
#      {"id": 8, "image_count": 272},
#      {"id": 9,  "image_count": 309},
#      {"id": 10, "image_count": 416},
#      {"id": 11, "image_count": 294},
#      {"id": 13,"image_count": 276},
#      {"id": 14, "image_count": 957},
#      {"id": 15, "image_count": 108},
#      {"id": 16, "image_count": 142},
#      {"id": 17, "image_count": 47},
#      {"id": 18,"image_count": 107},
#      {"id": 21,"image_count": 90},
#      {"id": 22, "image_count": 36},
#      {"id": 23, "image_count": 22},
#      {"id": 24, "image_count": 37},
#      {"id": 26, "image_count": 23},
#      {"id": 27,"image_count": 16},
#      {"id": 29, "image_count": 23},
#      {"id": 30,  "image_count": 2},
#      {"id": 32, "image_count": 3},
#      {"id": 4, "image_count": 0},
#      {"id": 5, "image_count": 0},
#      {"id": 7, "image_count": 0},
#      {"id": 12, "image_count": 0},
#      {"id": 19,"image_count": 0},
#      {"id": 20, "image_count": 0},
#      {"id": 25, "image_count": 0},
#      {"id": 28, "image_count": 0},
#      {"id": 31, "image_count": 0}
# ]

def _get_metadata(cat):
    # if cat == 'all':
    #     return _get_builtin_metadata('coco')
    count = None
    if cat == 'seen':
        id_to_name = {x['id']: x['name'] for x in categories_seen}
        count = category_image_count_train
    else:
        assert cat == 'unseen'
        id_to_name = {x['id']: x['name'] for x in categories_unseen}
        count = category_image_count_val

    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "class_image_count": count
    }


_CUSTOM_SPLITS_COCO = {
    "marinedet_v2_class_level_train": ("marinedet_v2", "marinedet_v2/annotations/class_level_train.json", 'seen'),
    "marinedet_v2_class_level_val": ("marinedet_v2", "marinedet_v2/annotations/class_level_val.json", 'unseen'),
}

for key, (image_root, json_file, cat) in _CUSTOM_SPLITS_COCO.items():
    # Assume pre-defined datasets live in `./datasets`.
    register_coco_instances(
        key,
        # _get_builtin_metadata('coco'),
        _get_metadata(cat), # empty metadata, it will be overwritten in load_coco_json() function
        # {},
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )