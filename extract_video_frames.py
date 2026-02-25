# extract_video_frames.py (多进程极速版)
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_one_video(args):
    """单个视频处理函数，用于多进程调用"""
    subject_id, trial_id, video_path, output_path = args

    # 路径兼容性检查
    video_file = os.path.join(video_path, f's{subject_id:02d}', f's{subject_id:02d}_trial{trial_id:02d}.avi')
    if not os.path.exists(video_file):
        video_file = os.path.join(video_path, f's{subject_id:02d}_trial{trial_id:02d}.avi')
    if not os.path.exists(video_file):
        return None

    save_path = os.path.join(output_path, f's{subject_id:02d}_trial{trial_id:02d}.npy')
    if os.path.exists(save_path):  # 跳过已存在的
        return 0

    cap = cv2.VideoCapture(video_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        # 在子进程中直接 Resize，减少主进程内存压力
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) > 0:
        np.save(save_path, np.array(frames, dtype=np.uint8))
        return len(frames)
    return 0


def extract_all_parallel(video_path, output_path, max_subjects=32, workers=8):
    os.makedirs(output_path, exist_ok=True)
    tasks = []

    print(f"🚀 启动 {workers} 个进程并行提取视频帧...")

    # 准备任务列表
    for subject_id in range(1, max_subjects + 1):
        for trial_id in range(1, 41):
            tasks.append((subject_id, trial_id, video_path, output_path))

    # 并行执行
    total_frames = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # 使用 tqdm 显示进度
        futures = [executor.submit(process_one_video, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Extracting"):
            res = future.result()
            if res: total_frames += res

    print(f"✅ 完成！总帧数: {total_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='/root/autodl-tmp/eeg/data/DEAP/face_video')
    parser.add_argument('--output_path', default='/root/autodl-tmp/eeg/data/DEAP/frames_cache')
    parser.add_argument('--workers', type=int, default=12, help='CPU核心数')
    args = parser.parse_args()

    extract_all_parallel(video_path=args.video_path, output_path=args.output_path, max_subjects=32, workers=args.workers)