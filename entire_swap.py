import cv2
from tqdm import tqdm
import argparse
from utils import *

parser = argparse.ArgumentParser(description="Entire face swapping")
parser.add_argument("--source", type=str, help="Name of source video")
parser.add_argument("--source_index", type=int, help="Index of source frame to use")
parser.add_argument("--target", type=str, help="Name of target video")
parser.add_argument("--target_start", type=int, help="Start index of target frame")
parser.add_argument("--output", type=str, help="Name of output video")
parser.add_argument("--frames", type=int, help="Number of frames")
parser.add_argument("--width", type=int, help="Width of output")
parser.add_argument("--height", type=int, help="Height of output")
args = parser.parse_args()

source = cv2.VideoCapture(args.source)
source_frames = []
while source.isOpened():
    ret, frame = source.read()
    if not ret:
        break
    source_frames.append(frame)
source.release()

target = cv2.VideoCapture(args.target)
target_frames = []
while target.isOpened():
    ret, frame = target.read()
    if not ret:
        break
    target_frames.append(cv2.resize(frame, (args.width, args.height)))
target.release()

source_frame_index = args.source_index
target_start = args.target_start
frames = args.frames

# Define the codec and create VideoWriter object
fourcc1 = cv2.VideoWriter_fourcc(*'MJPG')
out1 = cv2.VideoWriter(args.output, fourcc1, 20.0, (args.width, args.height))

source_frame = source_frames[source_frame_index]
source_rect = None
source_landmarks = None
faceRects = hogFaceDetector(source_frame, 0)
for faceRect in faceRects:
    source_rect = [faceRect.left(), faceRect.top(), faceRect.right(), faceRect.bottom()]
    source_landmarks = np.array([[p.x, p.y] for p in facePredictor(source_frame, faceRect).parts()])

if source_landmarks is None:
    print("Source video frame error")

center = np.zeros((2, 1))
center_last_frame = np.zeros((2, 1))

for i in tqdm(range(frames)):
    target_frame = target_frames[target_start + i]
    target_rect = None
    target_landmarks = None
    faceRects = hogFaceDetector(target_frame, 0)
    if len(faceRects) > 1:
        print("Multi Faces!")
        distance = []
        for faceRect in faceRects:
            target_rect = [faceRect.left(), faceRect.top(), faceRect.right(), faceRect.bottom()]
            center[0] = (target_rect[0] + target_rect[2]) / 2
            center[1] = (target_rect[1] + target_rect[3]) / 2
            distance.append((center[0] - center_last_frame[0]) ** 2 + (center[1] - center_last_frame[1]) ** 2)
        distance = np.array(distance)
        faceRect = faceRects[distance.argmin()]
    elif len(faceRects) == 0:
        faceRect = None
    else:
        faceRect = faceRects[0]

    if faceRect is None:
        out1.write(target_frame)
        continue
    else:
        target_landmarks = np.array([[p.x, p.y] for p in facePredictor(target_frame, faceRect).parts()])
        target_rect = [faceRect.left(), faceRect.top(), faceRect.right(), faceRect.bottom()]

    center_last_frame[0] = (target_rect[0] + target_rect[2]) / 2
    center_last_frame[1] = (target_rect[1] + target_rect[3]) / 2

    warp_source, mask = ImageMorphingTriangulation(source_frame, target_frame,
                                                   source_rect, source_landmarks, target_rect, target_landmarks)
    result = seamlessCloningPoisson(warp_source, target_frame, mask)
    result = skin_color_adjustment(result, target_frame, mask.astype(np.uint8))
    out1.write(result)

out1.release()
print("Done!")
