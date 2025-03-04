#概要
 このレポジトリは体幹CTから特定の臓器を抽出しCT値を測定するツールを作成しています。
 現状では体幹CTのデータを読み込んで特定の臓器を抽出するところまで行っています。2024,03,04現在
 またモデルをmonaiを使いたいところですが、リクエストが返ってこないので保留中です。

#仮想環境の有効化
 
conda activate monai2025

# 放射線科アプリの開始
monailabel start_server --app radiology --studies Task09_Spleen/imagesTr --conf models deepedit
