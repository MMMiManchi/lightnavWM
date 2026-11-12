"""Align a current entrance with a past camera from actual image geometry.

SIFT/RANSAC correspondences estimate a relative camera transform. Current
linear depth, when available, supports PnP and entrance-point reprojection;
otherwise only a well-supported pure-yaw rotation is accepted. Neither result
certifies physical place identity, traversability or a historical connection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


def wrapped(value):
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class EntranceAlignment:
    accepted: bool
    method: str
    bearing_offset: float = 0.0
    matches: int = 0
    inliers: int = 0
    median_reprojection_px: float | None = None
    reason: str = ""
    geometry_only_not_a_verified_connection: bool = True

    def transport(self, historical_bearing):
        if not self.accepted:
            raise ValueError("unverified camera alignment cannot transport directions")
        return wrapped(historical_bearing+self.bearing_offset)


class EntranceAligner:
    def __init__(self, *, hfov_degrees=120, minimum_inliers=16, reprojection_px=3.0):
        cv2.setNumThreads(1)
        self.hfov_degrees = hfov_degrees
        self.minimum_inliers = minimum_inliers
        self.reprojection_px = reprojection_px
        self.detector = cv2.SIFT_create(nfeatures=2500)
        self._descriptors = {}

    def intrinsics(self, shape):
        h,w = shape[:2]
        focal = w/(2*math.tan(math.radians(self.hfov_degrees)/2))
        return np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]],dtype=np.float64)

    def _features(self, pixels):
        import hashlib

        key = hashlib.sha256(pixels.tobytes()).hexdigest()
        if key not in self._descriptors:
            gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
            points,descriptors=self.detector.detectAndCompute(gray,None)
            self._descriptors[key]=(np.array([p.pt for p in points],dtype=np.float64),descriptors)
        return self._descriptors[key]

    def estimate(self, source_rgb, current_rgb, *, entry_point, current_depth=None):
        source,current=np.asarray(source_rgb),np.asarray(current_rgb)
        if source.shape!=current.shape or source.ndim!=3 or source.shape[-1]!=3:
            return EntranceAlignment(False,'none',reason='incompatible_camera_images')
        source_points,source_desc=self._features(source)
        current_points,current_desc=self._features(current)
        if source_desc is None or current_desc is None or min(len(source_desc),len(current_desc))<self.minimum_inliers:
            return EntranceAlignment(False,'none',reason='insufficient_visual_features')
        matcher=cv2.BFMatcher(cv2.NORM_L2)
        forward=matcher.knnMatch(source_desc,current_desc,k=2)
        reverse=matcher.knnMatch(current_desc,source_desc,k=2)
        reverse_map={m.queryIdx:m.trainIdx for pair in reverse if len(pair)==2
                     for m,n in [pair] if m.distance<.72*n.distance}
        good=[m for pair in forward if len(pair)==2 for m,n in [pair]
              if m.distance<.72*n.distance and reverse_map.get(m.trainIdx)==m.queryIdx]
        if len(good)<self.minimum_inliers:
            return EntranceAlignment(False,'none',matches=len(good),reason='insufficient_mutual_matches')
        return self.from_correspondences(source_points[[m.queryIdx for m in good]],
            current_points[[m.trainIdx for m in good]],shape=current.shape,entry_point=entry_point,current_depth=current_depth)

    @staticmethod
    def coverage(points,shape):
        h,w=shape[:2]
        if len(points)==0:
            return 0.0
        span=np.ptp(points,axis=0)
        return float(span[0]*span[1]/(w*h))

    def from_correspondences(self, source, current, *, shape, entry_point, current_depth=None):
        source,current=np.asarray(source,dtype=np.float64),np.asarray(current,dtype=np.float64)
        count=len(source)
        if count<self.minimum_inliers or source.shape!=current.shape or source.shape[1:]!=(2,):
            return EntranceAlignment(False,'none',matches=count,reason='insufficient_correspondences')
        h,w=shape[:2]
        k=self.intrinsics(shape)
        uv=np.array([entry_point[0]*w/1000,entry_point[1]*h/1000],dtype=np.float64)
        if not (0<uv[0]<w and 0<uv[1]<h):
            return EntranceAlignment(False,'none',matches=count,reason='entrance_outside_image')
        bearing=math.atan((w/2-uv[0])/k[0,0])
        if current_depth is not None:
            depth=np.asarray(current_depth).squeeze()
            if depth.shape!=(h,w):
                raise ValueError('current depth must align with RGB')
            observed=[]
            for x,y in current:
                x,y=int(round(x)),int(round(y))
                patch=depth[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)]
                valid=patch[np.isfinite(patch)&(patch>1e-5)]
                observed.append(float(np.median(valid)) if len(valid) else np.nan)
            observed=np.array(observed)
            valid=np.isfinite(observed)
            if valid.sum()>=self.minimum_inliers:
                pixels=np.c_[current[valid],np.ones(valid.sum())]
                xyz=(np.linalg.inv(k)@pixels.T).T*observed[valid,None]
                try:
                    ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,source[valid],k,None,
                        iterationsCount=200,reprojectionError=self.reprojection_px,confidence=.995,flags=cv2.SOLVEPNP_EPNP)
                    if ok and inliers is not None and len(inliers)>=self.minimum_inliers:
                        positions=inliers[:,0]
                        cv2.solvePnP(xyz[positions],source[valid][positions],k,None,rvec,tvec,True,flags=cv2.SOLVEPNP_ITERATIVE)
                        projected,_=cv2.projectPoints(xyz[positions],rvec,tvec,k,None)
                        error=np.linalg.norm(projected[:,0]-source[valid][positions],axis=1)
                        rotation,_=cv2.Rodrigues(rvec)
                        x,y=int(round(uv[0])),int(round(uv[1]))
                        patch=depth[max(0,y-2):min(h,y+3),max(0,x-2):min(w,x+3)]
                        values=patch[np.isfinite(patch)&(patch>1e-5)]
                        if len(values) and self.coverage(current[valid][positions],shape)>.05:
                            point=np.linalg.inv(k)@np.r_[uv,1.]*float(np.median(values))
                            old=rotation@point+tvec[:,0]
                            if old[2]>1e-5 and np.median(error)<=self.reprojection_px:
                                old_bearing=math.atan2(-old[0],old[2])
                                if abs(old_bearing)<math.radians(self.hfov_degrees/2):
                                    return EntranceAlignment(True,'current_depth_pnp',wrapped(bearing-old_bearing),
                                        count,len(positions),float(np.median(error)),'entrance_point_reprojected')
                except cv2.error:
                    pass
        # Pure rotation fallback rejects translation/parallax that cannot be
        # explained by one yaw, rather than widening the old direction gate.
        old_angle=np.arctan((w/2-source[:,0])/k[0,0])
        new_angle=np.arctan((w/2-current[:,0])/k[0,0])
        differences=np.arctan2(np.sin(new_angle-old_angle),np.cos(new_angle-old_angle))
        rays=(np.linalg.inv(k)@np.c_[source,np.ones(count)].T).T
        best=None
        for delta in differences[::max(1,count//80)]:
            cosine,sine=math.cos(delta),math.sin(delta)
            rotated=rays@np.array([[cosine,0,sine],[0,1,0],[-sine,0,cosine]])
            projected=(k@rotated.T).T
            error=np.linalg.norm(projected[:,:2]/projected[:,2,None]-current,axis=1)
            inliers=(error<self.reprojection_px)&(rotated[:,2]>0)
            if best is None or inliers.sum()>best[0].sum():
                best=(inliers,delta,error)
        inliers,delta,error=best
        if (inliers.sum()<self.minimum_inliers or inliers.mean()<.55
                or self.coverage(source[inliers],shape)<.08 or self.coverage(current[inliers],shape)<.08):
            return EntranceAlignment(False,'yaw_ransac',matches=count,inliers=int(inliers.sum()),reason='rotation_not_supported')
        delta=float(np.median(differences[inliers]))
        return EntranceAlignment(True,'yaw_ransac',delta,count,int(inliers.sum()),float(np.median(error[inliers])),
                                 'pure_yaw_supported_translation_not_claimed')


class ImageDirectionTransport:
    """An episode/frame-bound adapter for strict historical-direction queries."""

    def __init__(self,images,aligner=None):
        self.images=images
        self.aligner=aligner or EntranceAligner()
        self.current_depth=None
        self.last_trace=[]

    def for_branch(self,current_reference,branch):
        from dataclasses import asdict

        point=branch.entry_point
        if point is None:
            if branch.image_region is None:
                point=(500,700)
            else:
                x1,y1,x2,y2=branch.image_region
                point=((x1+x2)/2,y2-.1*(y2-y1))

        def transform(source_reference,old_direction):
            estimate=self.aligner.estimate(np.asarray(self.images.get(source_reference)),
                np.asarray(self.images.get(current_reference)),entry_point=point,current_depth=self.current_depth)
            self.last_trace.append({'source_observation':source_reference,'current_observation':current_reference,
                                    'candidate_id':branch.candidate_id,**asdict(estimate)})
            if not estimate.accepted:
                return None
            bearing=estimate.transport(math.atan2(*old_direction))
            return np.array([math.sin(bearing),math.cos(bearing)],dtype=np.float32),asdict(estimate)
        return transform
