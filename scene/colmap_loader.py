#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import collections
import struct

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    # "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
    "Image", ["id", "qvec", "tvec", "name"]
    )
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])
# qw qx qy qz  ##colmap format
# 0  1  2  3

# qx qy qz qw  ##our format
# 0  1  2  3

def qvec2rotmat(qvec):
    qx = qvec[0]
    qy = qvec[1]
    qz = qvec[2]
    qw = qvec[3]
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2,
         2 * qx * qy - 2 * qw * qz,
         2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz,
         1 - 2 * qx**2 - 2 * qz**2,
         2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy,
         2 * qy * qz + 2 * qw * qx,
         1 - 2 * qx**2 - 2 * qy**2]])

# def qvec2rotmat(qvec):
#     return np.array([
#         [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
#          2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
#          2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
#         [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
#          1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
#          2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
#         [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
#          2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
#          1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)
    
def read_points3D_binary(path_to_model_file):
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_intrinsics_text(path):
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                
                #TODO: what is the size? 
                #params(0)=fx, #param(1)=fy
                # cameras[camera_id] = Camera(id=camera_id, model=model,width=width, height=height,params=params)
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_extrinsics_text(path):  
    images = {}  
    with open(path, "r") as fid:  
        for line in fid:  
            line = line.strip()  
            if len(line) > 0 and line[0] != "#":  
                elems = line.split()  
                image_id = int(elems[0])  
                qvec = np.array(tuple(map(float, elems[4:8])))  
                tvec = np.array(tuple(map(float, elems[1:4])))   
                image_name = elems[8]  
                images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,name=image_name)  
    return images  