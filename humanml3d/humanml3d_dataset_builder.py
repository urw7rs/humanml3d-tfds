"""humanml3d dataset."""

import os
from pathlib import Path

import numpy as np
import torch

import tensorflow_datasets as tfds

import humanml3d_utils
from humanml3d_utils.core import AMASSBodyModel

import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


ex_fps = 20

base_url = "https://raw.githubusercontent.com/EricGuo5513/HumanML3D/ab5b332c3148ec669da4c55ad119e0d73861b867"


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for humanml3d dataset."""

    VERSION = tfds.core.Version("1.4.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "Generates cropped motions from tags.",
        "1.2.0": "deduplicates full motions.",
        "1.4.0": "return raw data.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Register into https://amass.is.tue.mpg.de to download amass data. Place the
    files in `manual_dir/`.
    """
    URLS = {
        "index.csv": base_url + "index.csv",
        "all.txt": base_url + "HumanML3D/all.txt",
        "text.zip": "https://raw.githubusercontent.com/EricGuo5513/HumanML3D/ab5b332c3148ec669da4c55ad119e0d73861b867/HumanML3D/texts.zip",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "motion": tfds.features.Tensor(shape=(None, 263), dtype=np.float32),
                    "caption": tfds.features.Text(),
                    "tokens": tfds.features.Text(),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("motion", "text"),  # Set to `None` to disable
            homepage="https://github.com/EricGuo5513/HumanML3D",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        manual_dir: Path = Path(dl_manager._manual_dir)
        extract_dir = manual_dir / "raw"
        humanact12_path = dl_manager.download(f"{base_url}/pose_data/humanact12.zip")
        index_path = dl_manager.download(f"{base_url}/index.csv")
        text_path = dl_manager.download_and_extract(f"{base_url}/HumanML3D/texts.zip")

        train, test, val, train_val = [
            dl_manager.download(f"{base_url}/HumanML3D/{name}")
            for name in ["train.txt", "test.txt", "val.txt", "train_val.txt"]
        ]

        smpl_path, dmpl_path = humanml3d_utils.extract_smpl_files(
            manual_dir, extract_dir
        )

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        body_model = AMASSBodyModel(smpl_path, dmpl_path).to(device)

        amass_paths = humanml3d_utils.extract_amass_files(manual_dir, extract_dir)
        humanact_paths = humanml3d_utils.load_humanact12(humanact12_path, extract_dir)

        positions = humanml3d_utils.to_positions(
            amass_paths,
            body_model,
            device,
        )
        positions.extend(humanact_paths)

        pose_representation = humanml3d_utils.motion_representation(
            humanml3d_utils.flip_left_right(
                humanml3d_utils.format_poses(
                    positions, root=extract_dir, index_path=index_path, fps=20
                )
            )
        )

        for array, path in pose_representation:
            path = manual_dir / "joint_vecs" / path.name
            os.makedirs(path.parent, exist_ok=True)
            np.save(path, array)

        for file in text_path.glob("*/*.txt"):
            text = file.read_text()

            path = manual_dir / "text" / file.name
            os.makedirs(path.parent, exist_ok=True)
            path.write_text(text)

        train, test, val, train_val = [
            humanml3d_utils.load_splits(
                split, manual_dir / "joint_vecs", manual_dir / "text"
            )
            for split in [train, test, val, train_val]
        ]

        return {
            "train": self._generate_examples(train),
            "val": self._generate_examples(val),
            "train_val": self._generate_examples(train_val),
            "test": self._generate_examples(test),
        }

    def _generate_examples(self, path_pairs):
        """Yields examples."""

        i = 0
        for joint_path, text_path in path_pairs:
            data = np.load(joint_path)
            text = text_path.read_text()

            caption_token_pairs = []
            for line in text.splitlines():
                caption, tokens, f_tag, to_tag = line.strip().split("#")

                f_tag = float(f_tag)
                to_tag = float(to_tag)

                # some tags have the text nan in them
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                if f_tag == 0.0 and to_tag == 0.0:
                    motion = data
                    caption_token_pairs.append((caption, tokens))
                else:
                    motion = data[int(f_tag * ex_fps) : int(to_tag * ex_fps)]

                    yield str(i), {
                        "motion": motion,
                        "caption": caption,
                        "tokens": tokens,
                    }
                    i += 1

            if len(caption_token_pairs) > 0:
                if len(caption_token_pairs) == 1:
                    caption, tokens = caption_token_pairs[0]
                else:
                    caption, tokens = zip(*caption_token_pairs)

                yield str(i), {
                    "motion": data,
                    "caption": "\n".join(caption),
                    "tokens": "\n".join(tokens),
                }
                i += 1
