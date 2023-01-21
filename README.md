# Face Swapping in Videos

This model is to automatically detect faces in videos and then swap them. The pipeline contains four major approaches: face detection and identification, feature extraction, face swapping which includes warping and blending, as well as multi-frame manipulation. There are two different methods for the face swapping: entire swap where the entire source face together with all its features and emotion is replaced to the target; and separate swap where only part of the facial features from the source are replaced to the target with the emotion on the target face still kept.

To run the code, use command `python3 entire_swap.py --source --source_index --target --target_start --output --frames --width --height` or `python3 separate_swap.py --source --source_index --target --target_start --output --frames --width --height`. The flags are described below:
+ `--source`: Name of source video
+ `--source_index`: Index of source frame to use
+ `--target`: Name of target video
+ `--target_start`: Start index of target frame
+ `--output`: Name of output video
+ `--frames`: Number of frames
+ `--width`: Width of output
+ `--height`: Height of output

Example command: `python3 entire_swap.py --source "sources/MrRobot.mp4" --source_index 50 --target "targets/FrankUnderwood.mp4" --target_start 0 --output "test1.mp4" --frames 200 --width 640 --height 360`