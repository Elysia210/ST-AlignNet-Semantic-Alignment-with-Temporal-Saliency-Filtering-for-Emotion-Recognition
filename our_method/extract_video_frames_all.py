import os
import cv2
from tqdm import tqdm

def extract_frames_from_video(video_path, output_dir, resize=(224, 224), fps_limit=None):
    """
    将视频提取为图像帧并保存
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Cannot open video: {video_path}")
        return

    frame_count = 0
    saved_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval = int(fps // fps_limit) if fps_limit else 1
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()

def process_all_trials(video_root_dir, output_root_dir, subject_range=range(1, 23), resize=(224, 224), fps_limit=25):
    """
    批量处理所有受试者 s01~s22 的 40 段 trial 视频
    """
    for subj_id in subject_range:
        subj_str = f"s{subj_id:02d}"
        subj_input_dir = os.path.join(video_root_dir, subj_str)
        subj_output_dir = os.path.join(output_root_dir, subj_str)
        os.makedirs(subj_output_dir, exist_ok=True)

        print(f"\n📂 Processing subject: {subj_str}")
        for trial_id in range(1, 41):
            video_file = f"{subj_str}_trial{trial_id:02d}.avi"
            video_path = os.path.join(subj_input_dir, video_file)
            trial_output_dir = os.path.join(subj_output_dir, f"{subj_str}_trial{trial_id:02d}")
            if not os.path.exists(video_path):
                print(f"⚠️  Missing: {video_file}")
                continue
            extract_frames_from_video(video_path, trial_output_dir, resize, fps_limit)
            print(f"✅ Saved: {video_file} → {trial_output_dir}")

if __name__ == "__main__":
    # 请修改为你实际的路径：
    input_root = r"/root/autodl-tmp/eeg/data/DEAP/face_video"           # 视频根目录（s01 ~ s22）
    output_root = r"/root/autodl-tmp/eeg/data/DEAP/face_frames"   # 输出帧图目录

    process_all_trials(video_root_dir=input_root, output_root_dir=output_root)
