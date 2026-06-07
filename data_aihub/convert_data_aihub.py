import json
import numpy as np
from pathlib import Path

# 설정
KEYPOINT_ROOT    = Path("keypoint/01")
MORPHEME_ROOT    = Path("morpheme/morpheme/01")
OUTPUT_ROOT      = Path("dataset_words")
SEQ_LEN          = 30   # train_lstm_aihub와 동일하게 시퀀스 30 
MIN_VALID_FRAMES = 10   

OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

def extract_hand_landmarks(keypoint_data: dict) -> np.ndarray | None:
    people = keypoint_data.get("people", [])
    if not people:
        return None
    person = people[0] if isinstance(people, list) else people
    
    right_points = person.get("hand_right_keypoints_3d", [])
    left_points = person.get("hand_left_keypoints_3d", [])

    # 유효성 검사 (데이터가 존재하고 전부 0이 아닌지)
    is_right_valid = len(right_points) >= 84 and np.abs(np.array(right_points)).sum() > 1e-6
    is_left_valid = len(left_points) >= 84 and np.abs(np.array(left_points)).sum() > 1e-6
    
    # 양손이 모두 검출되지 않은 프레임은 건너뜀
    if not is_right_valid and not is_left_valid:
        return None

    # 1. 오른손 처리 (21개 키포인트 * 3차원 = 63차원)
    if is_right_valid:
        r_coords = [right_points[i:i+3] for i in range(0, 84, 4)]
        r_arr = np.array(r_coords, dtype=np.float32)
        r_wrist = r_arr[0].copy()
        r_arr -= r_wrist  # 오른손 손목 기준 영점 조절
    else:
        r_arr = np.zeros((21, 3), dtype=np.float32)

    # 2. 왼손 처리 (21개 키포인트 * 3차원 = 63차원)
    if is_left_valid:
        l_coords = [left_points[i:i+3] for i in range(0, 84, 4)]
        l_arr = np.array(l_coords, dtype=np.float32)
        l_wrist = l_arr[0].copy()
        l_arr -= l_wrist  # 왼손 손목 기준 영점 조절
    else:
        l_arr = np.zeros((21, 3), dtype=np.float32)

    # 양손 데이터를 결합하여 하나의 플래팅된 배열로 반환 (63 + 63 = 126차원)
    return np.vstack([r_arr, l_arr]).flatten()

def resample(frames: np.ndarray, target: int = SEQ_LEN) -> np.ndarray:
    idx = np.linspace(0, len(frames) - 1, target).astype(int)
    return frames[idx]

def process_clip(folder: Path, start: float, end: float, duration: float) -> np.ndarray | None:
    json_files = sorted(folder.glob("*_keypoints.json"))
    total_frames = len(json_files)
    if total_frames == 0:
        return None

    fps = total_frames / duration if duration > 0 else 25.0
    start_idx = max(0, int(start * fps))
    end_idx = min(total_frames - 1, int(end * fps))

    if end_idx <= start_idx:
        return None

    valid_frames = []
    for jf in json_files[start_idx: end_idx + 1]:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            lm = extract_hand_landmarks(data)
            if lm is not None:
                valid_frames.append(lm)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    if len(valid_frames) < MIN_VALID_FRAMES:
        return None
    return resample(np.array(valid_frames))

def parse_morpheme(json_path: Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    duration = raw.get("metaData", {}).get("duration", 0)
    segments = []
    for entry in raw.get("data", []):
        attrs = entry.get("attributes", [])
        if not attrs: continue
        word = attrs[0].get("name", "").strip()
        if not word: continue
        segments.append({
            "word": word,
            "start": entry.get("start", 0),
            "end": entry.get("end", duration),
            "duration": duration,
        })
    return segments

# MAIN

def main():
    morpheme_files = sorted(MORPHEME_ROOT.rglob("*.json"))
    print(f"morpheme 파일 총 {len(morpheme_files)}개 처리 시작\n") # 디버깅용

    seq_counter: dict[str, int] = {}
    total_saved   = 0
    total_skipped = 0

    for morph_path in morpheme_files:
        segments = parse_morpheme(morph_path)

        stem_name = morph_path.stem
        if stem_name.endswith("_morpheme"):
            folder_name = stem_name.replace("_morpheme", "")
        else:
            folder_name = stem_name
            
        kp_folder = KEYPOINT_ROOT / folder_name

        for seg in segments:
            word = seg["word"]

            if not kp_folder.is_dir():
                total_skipped += 1
                continue

            seq = process_clip(kp_folder, seg["start"], seg["end"], seg["duration"])

            if seq is None:
                total_skipped += 1
                continue

            # 저장 경로 생성
            out_dir = OUTPUT_ROOT / word
            out_dir.mkdir(exist_ok=True, parents=True)

            idx = seq_counter.get(word, 0)
            np.save(out_dir / f"seq_{idx:04d}.npy", seq)

            seq_counter[word] = idx + 1
            total_saved += 1

    # 결과 요약
    print("-" * 50)
    print(f"변환 완료 : {total_saved}개 시퀀스")
    print(f"스킵      : {total_skipped}개 (폴더 없음 / 유효 프레임 부족)")
    print(f"추출 단어 : {len(seq_counter)}개\n")

if __name__ == "__main__":
    main()