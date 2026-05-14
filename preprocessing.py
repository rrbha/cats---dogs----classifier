import zipfile # library for handling .zip archive files
# a context manager to open the archive and extract its contents for data preparation
with zipfile.ZipFile("Downloads2.zip") as zip_ref:
    zip_ref.extractall("dataset") # decompress all images and subfolders into a directory named 'dataset'
