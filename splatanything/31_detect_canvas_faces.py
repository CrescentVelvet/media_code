"""Detect nose and face on the exact RGB canvas used by each diagnostic stage."""
import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out",type=Path)
    out=ap.parse_args().out
    faces={}
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,
                                         min_detection_confidence=.25) as model:
        for row in json.loads((out/"face_requests.json").read_text()):
            x,y,xx,yy=row["bbox"]
            crop=np.asarray(Image.open(row["file"]).convert("RGB"))[y:yy,x:xx]
            for scale in (1,2,.75):
                result=model.process(cv2.resize(crop,None,fx=scale,fy=scale))
                if result.multi_face_landmarks:
                    faces[row["key"]]=[[x+p.x*(xx-x),y+p.y*(yy-y)]
                                       for p in result.multi_face_landmarks[0].landmark]
                    break
    (out/"landmarks_canvas.json").write_text(json.dumps(faces))
    print("Detected:", list(faces))


if __name__=="__main__":
    main()
