"""humanml3d dataset."""

import os
from pathlib import Path
import itertools

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

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Register into https://amass.is.tue.mpg.de to get the data. Place the `data.zip`
    file in the `manual_dir/`.
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
                    "text": tfds.features.Text(),
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
        extracted_dir = manual_dir / "extracted"

        (
            male_bm_path,
            female_bm_path,
            male_dmpl_path,
            female_dmpl_path,
        ) = humanml3d_utils.extract_smpl_files(manual_dir, extracted_dir)

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        body_model = AMASSBodyModel(
            male_bm_path,
            female_bm_path,
            male_dmpl_path,
            female_dmpl_path,
        ).to(device)

        extracted_paths = humanml3d_utils.extract_amass_files(
            dl_manager.manual_dir, extracted_dir
        )

        amass_positions = humanml3d_utils.positions(extracted_paths, body_model, device)

        index_path = dl_manager.download(f"{base_url}/index.csv")

        humanact_path = dl_manager.download(f"{base_url}/pose_data/humanact12.zip")
        humanact_positions = humanml3d_utils.extract_humanact12(
            humanact_path, extracted_dir
        )

        positions = itertools.chain(amass_positions, humanact_positions)

        pose_representation = humanml3d_utils.motion_representation(
            humanml3d_utils.flip_left_right(
                humanml3d_utils.format(positions, extracted_dir, index_path)
            )
        )

        for array, path in pose_representation:
            path = manual_dir / "joint_vecs" / path.name
            os.makedirs(path.parent, exist_ok=True)
            np.save(path, array)

        text_path = dl_manager.download_and_extract(f"{base_url}/HumanML3D/texts.zip")
        for file in text_path.glob("*/*.txt"):
            text = file.read_text()

            path = dl_manager.manual_dir / "text" / file.name
            os.makedirs(path.parent, exist_ok=True)
            path.write_text(text)

        train, test, val, train_val = [
            humanml3d_utils.load_splits(
                dl_manager.download(f"{base_url}/HumanML3D/{name}"),
                manual_dir / "joint_vecs",
                manual_dir / "text",
            )
            for name in ["train.txt", "test.txt", "val.txt", "train_val.txt"]
        ]

        # TODO(humanml3d): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(train),
            "val": self._generate_examples(val),
            "train_val": self._generate_examples(train_val),
            "test": self._generate_examples(test),
        }

    def _generate_examples(self, path_pairs):
        """Yields examples."""

        for i, (joint_path, text_path) in enumerate(path_pairs):
            data = np.load(joint_path)
            text = text_path.read_text()
            yield str(i), {"motion": data, "text": text}
