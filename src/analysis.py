import numpy as np

def compute_statistics(ct_image, seg_mask, target_label):
    """
    セグメンテーションマスクから、指定した臓器（target_label）に対応する領域のCT値統計量を計算します。
    """
    # 対象臓器のマスクを作成
    target_mask = (seg_mask == target_label)
    # 対象領域のCT値を抽出
    target_values = ct_image[target_mask]
    
    if target_values.size == 0:
        return None
    
    stats = {
        "mean": float(np.mean(target_values)),
        "std": float(np.std(target_values)),
        "min": float(np.min(target_values)),
        "max": float(np.max(target_values)),
        "num_voxels": int(target_values.size)
    }
    return stats