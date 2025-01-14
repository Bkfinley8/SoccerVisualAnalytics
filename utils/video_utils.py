import cv2

def read_video(path):
    print("... Reading Video ...")
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(video_frames,output_path):
    print("... Saving Video ...")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 24, (video_frames[0].shape[1], video_frames[0].shape[0]))
    for frame in video_frames:
        out.write(frame)
    out.release()