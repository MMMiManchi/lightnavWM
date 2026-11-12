import math

import cv2
import numpy as np

from experiments.world_model.entrance_alignment import EntranceAligner


def test_image_alignment_recovers_camera_yaw_without_pose_or_graph():
    rng=np.random.default_rng(7)
    image=rng.integers(0,256,(270,480,3),dtype=np.uint8)
    image=cv2.GaussianBlur(image,(3,3),0)
    aligner=EntranceAligner()
    k=aligner.intrinsics(image.shape)
    delta=math.radians(12)
    rotation=np.array([[math.cos(delta),0,-math.sin(delta)],[0,1,0],[math.sin(delta),0,math.cos(delta)]])
    current=cv2.warpPerspective(image,k@rotation@np.linalg.inv(k),(480,270))
    result=aligner.estimate(image,current,entry_point=(500,700))
    assert result.accepted and abs(result.bearing_offset-delta)<math.radians(1)
    assert abs(result.transport(-.3)-(-.3+delta))<math.radians(1)
    assert result.geometry_only_not_a_verified_connection


def test_unrelated_views_are_unknown_instead_of_reusing_old_directions():
    rng=np.random.default_rng(8)
    aligner=EntranceAligner()
    result=aligner.estimate(rng.integers(0,256,(270,480,3),dtype=np.uint8),
                           rng.integers(0,256,(270,480,3),dtype=np.uint8),entry_point=(500,700))
    assert not result.accepted


def test_current_depth_pnp_handles_translation_and_reprojects_entrance():
    rng=np.random.default_rng(10)
    aligner=EntranceAligner()
    k=aligner.intrinsics((270,480,3))
    current=rng.uniform([70,40],[410,230],(100,2))
    depth=np.zeros((270,480),dtype=np.float32)
    distances=rng.uniform(3,8,100)
    xyz=(np.linalg.inv(k)@np.c_[current,np.ones(100)].T).T*distances[:,None]
    rvec=np.array([0.,.12,0.])
    tvec=np.array([.3,0,.15])
    source,_=cv2.projectPoints(xyz,rvec,tvec,k,None)
    for (x,y),z in zip(current,distances,strict=True):
        depth[int(round(y)),int(round(x))]=z
    depth[186:193,237:244]=5
    result=aligner.from_correspondences(source[:,0],current,shape=(270,480,3),entry_point=(500,700),current_depth=depth)
    assert result.accepted and result.method=='current_depth_pnp'
    rotation,_=cv2.Rodrigues(rvec)
    point=rotation@(np.linalg.inv(k)@np.array([240,189,1])*5)+tvec
    expected=-math.atan2(-point[0],point[2])
    assert abs(result.bearing_offset-expected)<.02
