def load_dataset():
    # Dataset does not fit locally so download from source
    import os
    import requests
    import tarfile
    import shutil

    def safe_filter(tarinfo, extract_path):
        # Reject absolute paths or parent directory traversal
        if os.path.isabs(tarinfo.name) or ".." in tarinfo.name:
            return None
        return tarinfo  # keep original folder structure

    spanish_url = "https://zenodo.org/records/8220853/files/ChatSubs.tar.gz?download=1"
    spanish_filename = "ChatSubs.tar.gz"
    spanish_raw_path = "data/1-raw/spanish"

    print("Downloading Spanish dataset...")
    response = requests.get(spanish_url, stream=True)
    response.raise_for_status()  # Check if the download was successful

    os.makedirs(spanish_raw_path, exist_ok=True)

    with open(os.path.join(spanish_raw_path, spanish_filename), "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Download completed. Extracting files...")
    with tarfile.open(os.path.join(spanish_raw_path, spanish_filename), "r:gz") as tar:
        tar.extractall(path="data/1-raw/spanish/", filter=safe_filter)

    print("Extraction completed. Cleaning up...")
    os.remove(os.path.join(spanish_raw_path, spanish_filename))
    for item in os.listdir(os.path.join(spanish_raw_path, "open_subtitles_es")):
        shutil.move(
            os.path.join(spanish_raw_path, "open_subtitles_es", item),
            os.path.join(spanish_raw_path, item),
        )
    shutil.rmtree(os.path.join(spanish_raw_path, "open_subtitles_es"))
    shutil.rmtree(os.path.join(spanish_raw_path, "open_subtitles_ca"))
    shutil.rmtree(os.path.join(spanish_raw_path, "open_subtitles_eu"))
    shutil.rmtree(os.path.join(spanish_raw_path, "open_subtitles_gl"))
    print("Spanish dataset downloaded and extracted successfully.")
