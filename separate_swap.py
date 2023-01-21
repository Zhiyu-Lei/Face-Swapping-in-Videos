import cv2
from tqdm import tqdm
import argparse
from utils import *
import transfer

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
fps = target.get(cv2.CAP_PROP_FPS)
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
fourcc2 = cv2.VideoWriter_fourcc(*'MJPG')
out2 = cv2.VideoWriter(args.output, fourcc2, fps, (args.width, args.height))

source_frame = source_frames[source_frame_index]
source_frame = cv2.resize(source_frame, (600, source_frame.shape[0] * 600 // source_frame.shape[1]))
landmarks1 = transfer.get_face_landmarks(source_frame, hogFaceDetector, facePredictor)  # 68_face_landmarks

if landmarks1 is None:
    print('No face detected')
source_frame_mask = transfer.get_face_mask(source_frame, landmarks1)

for i in tqdm(range(frames)):
    landmarks2 = transfer.get_face_landmarks(target_frames[target_start + i], hogFaceDetector, facePredictor)  # 68_face_landmarks

    if landmarks2 is not None:
        frame_swapped = transfer.transfer_img(source_frame, target_frames[target_start + i], source_frame_mask, landmarks1, landmarks2)
        out2.write(frame_swapped)
    else:
        out2.write(target_frames[target_start + i])

out2.release()
print("Done!")
