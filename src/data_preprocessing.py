import os
import nibabel as nib
import SimpleITK as sitk

def load_nifti_image(file_path):
    """
    NIFTI形式の画像を読み込み、numpy配列とアフィン行列を返します。
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

def load_dicom_series(directory_path):
    """
    指定したディレクトリ内のDICOMファイルを読み込み、3D画像のnumpy配列とSimpleITK Imageを返します。
    DICOMシリーズは複数のファイルから構成されるため、SimpleITKのImageSeriesReaderを用いてまとめて読み込みます。
    """
    # DICOMファイルのリストを取得
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory_path)
    
    if not dicom_names:
        raise FileNotFoundError(f"DICOMファイルがディレクトリ {directory_path} に見つかりませんでした。")
    
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # 画像をnumpy配列に変換（形状は [スライス数, 高さ, 幅]）
    data = sitk.GetArrayFromImage(image)
    
    # 必要に応じてアフィン変換行列を生成（SimpleITKでは直接のアフィンは取得できません）
    # ここでは、origin, spacing, directionから近似的な変換行列を構築するか、用途に合わせた別途対応が必要です。
    # 今回は簡略化のためNoneを返します。
    affine = None
    return data, image, affine

def save_nifti_image(data, affine, out_path):
    """
    numpy配列からNIFTI画像を作成し、指定パスに保存します。
    """
    if affine is None:
        # アフィン行列がない場合は、単にNifti1Imageを作成
        img = nib.Nifti1Image(data, affine=np.eye(4))
    else:
        img = nib.Nifti1Image(data, affine)
    nib.save(img, out_path)
