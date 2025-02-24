import torch
from monai.networks.nets import SegResNet

def load_model(model_path, device="cuda"):
    """
    事前学習済みモデルを読み込み、指定されたデバイスに配置します。
    """
    # 例としてSegResNetを使用しています。実際のモデルに合わせてパラメータを調整してください。
    model = SegResNet(spatial_dims=3, in_channels=1, out_channels=106)
    state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()


    return model

def run_segmentation(model, ct_image, device="cuda"):
    """
    CT画像（numpy配列）をモデルに入力し、セグメンテーションマスク（numpy配列）を出力します。
    ※ 実際の前処理（リサンプリングや正規化など）はデータに応じて実装してください。
    """
    # 例：CT画像をfloat32型に変換し、形状を (1, 1, D, H, W) に整形
    img_tensor = torch.from_numpy(ct_image.astype("float32")).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model(img_tensor)  # 出力は (1, num_classes, D, H, W)
        # softmaxの後、argmaxにより最も高い確率のクラスを選択
        seg = torch.argmax(output, dim=1).squeeze(0)
    return seg.cpu().numpy()