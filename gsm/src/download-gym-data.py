from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="roman-bushuiev/GeMS",
    repo_type="dataset",
    allow_patterns="data/auxiliary/*",
    local_dir="../data/dreams_data/auxiliary/"
)