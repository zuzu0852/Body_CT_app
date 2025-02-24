import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch

import data_preprocessing as dp
import segmentation
import analysis
import utils

def main(config_path):
    # 設定ファイルの読み込み
    config = utils.load_config(config_path)
    
    model_path = config["model_path"]
    target_label = config["target_label"]
    input_path = config["input_image"]  # ここはDICOMシリーズが格納されたディレクトリ
    device = config.get("device", "cuda")
    input_format = config.get("input_format", "dcm")  # "dcm"または"nii"
    
    print("設定ファイルからパラメータを読み込みました:")
    print(f"  モデルパス: {model_path}")
    print(f"  対象ラベル: {target_label}")
    print(f"  入力パス: {input_path}")
    print(f"  入力形式: {input_format}")
    print(f"  デバイス: {device}")
    
    # 画像の読み込み（入力形式に応じて切り替え）
    if input_format.lower() == "nii":
        ct_image, affine = dp.load_nifti_image(input_path)
    elif input_format.lower() == "dcm":
        ct_image, sitk_image, affine = dp.load_dicom_series(input_path)
    else:
        raise ValueError("input_format は 'nii' または 'dcm' を指定してください。")
    
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
