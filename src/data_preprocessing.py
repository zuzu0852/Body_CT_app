import nibabel as nib

def load_nifti_image(file_path):
    """
    NIFTI形式の画像を読み込み、numpy配列とアフィン行列を返します。
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

def save_nifti_image(data, affine, out_path):
    """
    numpy配列からNIFTI画像を作成し、指定パスに保存します。
    """
    img = nib.Nifti1Image(data, affine)
    nib.save(img, out_path)