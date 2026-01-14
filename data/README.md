##  How to
- Find online or generate a Lerobot compatible dataset
- Use the HuggingFace link to the dataset or download and use a local version of the files located ideally in: `./data/<task-name>/<data-tag>`
- (opt upload the file on hf after generating the dataset)
```
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="<hf_user>/<hf_repo>", repo_type="dataset")
api.create_tag(repo_id="<hf_user>/<hf_repo>", tag="v2.0", repo_type="dataset")
api.upload_folder(folder_path="trajectories_converted", repo_id="<hf_user>/<hf_repo>", repo_type="dataset")
```
- Modify the `./configs/agents/Lerobot/<task-name>/lerobot_<agent>_<policy>_cfg.yaml` specifying the dataset and fill the `env_cfg` field accordingly to the lerobot dataset meta/info.json
