import os
import argparse
import torch

from src import data_preprocessing as dp
from src import segmentation
from src import analysis
from src import utils

def main(config_path):
    # 設定ファイルの読み込み
    config = utils.load_config(config_path)
    
    model_path = config["model_path"]
    target_label = config["target_label"]
    input_image_path = config["input_image"]
    device = config.get("device", "cpu")
    
    print("設定ファイルからパラメータを読み込みました:")
    print(f"  モデルパス: {model_path}")
    print(f"  対象ラベル: {target_label}")
    print(f"  入力画像パス: {input_image_path}")
    print(f"  デバイス: {device}")
    
    # CT画像の読み込み
    print("CT画像を読み込み中...")
    ct_image, affine = dp.load_nifti_image(input_image_path)
    
    # モデルの読み込み
    print("モデルを読み込み中...")
    model = segmentation.load_model(model_path, device=device)
    
    # セグメンテーションの実行
    print("セグメンテーションを実行中...")
    seg_mask = segmentation.run_segmentation(model, ct_image, device=device)
    
    # 対象臓器のCT値統計量を計算
    print("統計量を計算中...")
    stats = analysis.compute_statistics(ct_image, seg_mask, target_label)
    
    if stats is None:
        print("対象臓器が見つかりませんでした。")
    else:
        print("対象臓器のCT値統計量:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # セグメンテーションマスクを保存（例：NIFTI形式で保存）
    output_mask_path = os.path.join("data", "processed", "segmentation_mask.nii.gz")
    dp.save_nifti_image(seg_mask.astype("int16"), affine, output_mask_path)
    print(f"セグメンテーションマスクを {output_mask_path} に保存しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT画像のセグメンテーションおよび統計量計算アプリ")
    parser.add_argument("--config", type=str, default="config/config.json", help="設定ファイルのパス")
    args = parser.parse_args()
    
    main(args.config)