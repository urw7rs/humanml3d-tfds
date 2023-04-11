"""humanml3d dataset."""

import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import spacy

import tensorflow_datasets as tfds

from humanml3d_utils.files import extract_tar, extract_files, amass_files
from humanml3d_utils.raw_pose_processing import AMASSBodyModel, swap_left_right
from humanml3d_utils.skeletons import skeleton_factory, Skeleton
import humanml3d_utils.motion_representation as mr_utils


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
        # TODO(humanml3d): Specifies the tfds.core.DatasetInfo object
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
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(humanml3d): Downloads the data and defines the splits
        # path = dl_manager.download_and_extract("https://todo-data-url")
        root: Path = Path(dl_manager._manual_dir)

        smpl_dir = root / "smpl_data"
        pose_dir = root / "pose_data"
        smpl_model_dir = root / "smpl_models"

        smpl_dir.mkdir(exist_ok=True)
        pose_dir.mkdir(exist_ok=True)
        smpl_model_dir.mkdir(exist_ok=True)

        amass_paths = [root / path for path in amass_files]

        humanact_path = dl_manager.download_and_extract(
            f"{base_url}/pose_data/humanact12.zip"
        )

        for path in humanact_path.glob("*/*/*.npy"):
            data = np.load(path)
            path = pose_dir / path.relative_to(humanact_path)
            os.makedirs(str(path.parent), exist_ok=True)
            np.save(path, data)

        extract_tar(root / "smplh.tar.xz", smpl_model_dir / "smplh", "xz")
        extract_tar(root / "dmpls.tar.xz", smpl_model_dir / "dmpls", "xz")
        extract_files(amass_paths, smpl_dir, "bz2")

        male_bm_path = smpl_model_dir / "smplh/male/model.npz"
        male_dmpl_path = smpl_model_dir / "dmpls/male/model.npz"

        female_bm_path = smpl_model_dir / "smplh/female/model.npz"
        female_dmpl_path = smpl_model_dir / "dmpls/female/model.npz"

        num_betas = 10  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        body_model = AMASSBodyModel(
            male_bm_path,
            female_bm_path,
            male_dmpl_path,
            female_dmpl_path,
            num_betas,
            num_dmpls,
        ).to(device)

        index_path = dl_manager.download(f"{base_url}/index.csv")
        index_file = pd.read_csv(index_path)

        raw_offsets, kinematic_chain = skeleton_factory("humanml3d", device)

        for path in smpl_dir.glob("**/*.npz"):
            bdata = np.load(path, allow_pickle=True)

            fps = int(bdata.get("mocap_framerate", 0))
            frame_number = bdata.get("trans", None)
            if fps == 0 or frame_number is None:
                continue
            else:
                frame_number = frame_number.shape[0]

            gender = bdata["gender"]

            def as_tensor(key):
                return torch.tensor(bdata[key], dtype=torch.float32, device=device)

            keys = ["poses", "betas", "trans"]
            bdata = {key: as_tensor(key) for key in keys}

            poses = bdata["poses"]
            betas = bdata["betas"]
            trans = bdata["trans"]

            pose_seq = body_model(trans, gender, fps, poses, betas)
            path = pose_dir / path.relative_to(smpl_dir).with_suffix(".npy")
            os.makedirs(path.parent, exist_ok=True)
            np.save(path, pose_seq.cpu().numpy())

        joint_dir = root / "joint"
        new_joint_dir = root / "new_joint"
        joint_vec_dir = root / "joint_vec"

        joint_dir.mkdir(exist_ok=True)
        new_joint_dir.mkdir(exist_ok=True)
        joint_vec_dir.mkdir(exist_ok=True)

        fps = 20
        for path, start_frame, end_frame, new_name in zip(
            index_file["source_path"],
            index_file["start_frame"],
            index_file["end_frame"],
            index_file["new_name"],
        ):
            path = pose_dir / Path(path).relative_to("pose_data")

            pose_seq = np.load(path)

            if "humanact12" not in str(path):
                if "Eyes_Japan_Dataset" in str(path):
                    pose_seq = pose_seq[3 * fps :]
                if "MPI_HDM05" in str(path):
                    pose_seq = pose_seq[3 * fps :]
                if "TotalCapture" in str(path):
                    pose_seq = pose_seq[1 * fps :]
                if "MPI_Limits" in str(path):
                    pose_seq = pose_seq[1 * fps :]
                if "Transitions_mocap" in str(path):
                    pose_seq = pose_seq[int(0.5 * fps) :]
                pose_seq = pose_seq[start_frame:end_frame]
                pose_seq[..., 0] *= -1

            pose_seq_m = swap_left_right(pose_seq)

            np.save(joint_dir / new_name, pose_seq)
            np.save(joint_dir / ("M" + new_name), pose_seq_m)

        raw_offsets, kinematic_chain = skeleton_factory("humanml3d", device)
        skeleton = Skeleton(raw_offsets, kinematic_chain, "cpu")

        target_skeleton = np.load(joint_dir / "000021.npy")
        target_skeleton = target_skeleton.reshape(len(target_skeleton), -1, 3)
        target_skeleton = torch.from_numpy(target_skeleton).cpu()

        target_offsets = skeleton.get_offsets_joints(target_skeleton[0])

        for path in joint_dir.glob("**/*.npy"):
            source_data = np.load(path)[:, : skeleton.njoints()]

            if source_data.shape[0] == 1:
                print(path)
                continue

            data, ground_positions, positions, l_velocity = mr_utils.process_file(
                raw_offsets, kinematic_chain, source_data, 0.002, target_offsets
            )
            rec_ric_data = (
                mr_utils.recover_from_ric(
                    torch.from_numpy(data).unsqueeze(0).float().to(device),
                    skeleton.njoints(),
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            np.save(new_joint_dir / path.name, rec_ric_data)
            np.save(joint_vec_dir / path.name, data)

        text_path = dl_manager.download_and_extract(f"{base_url}/HumanML3D/texts.zip")
        for file in text_path.glob("*/*.txt"):
            text = file.read_text()
            path = dl_manager.manual_dir / "text" / file.name
            os.makedirs(path.parent, exist_ok=True)
            path.write_text(text)

        def load(path):
            names = path.read_text().splitlines()
            joint_paths = [joint_vec_dir / f"{name}.npy" for name in names]
            text_paths = [
                dl_manager.manual_dir / "text" / f"{name}.txt" for name in names
            ]
            return list(zip(joint_paths, text_paths))

        train, test, val, train_val = [
            load(dl_manager.download(f"{base_url}/HumanML3D/{name}"))
            for name in ["train.txt", "test.txt", "val.txt", "train_val.txt"]
        ]

        breakpoint()

        # TODO(humanml3d): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(train),
            "val": self._generate_examples(val),
            "train_val": self._generate_examples(train_val),
            "test": self._generate_examples(test),
        }

    def _generate_examples(self, path_pairs):
        """Yields examples."""
        # TODO(humanml3d): Yields (key, example) tuples from the dataset
        breakpoint()

        for joint_path, text_path in path_pairs:
            data = np.load(joint_path)
            text = text_path.read_text()
            yield "key", {"motion": data, "text": text}


nlp = spacy.load("en_core_web_sm")


def process_text(sentence):
    sentence = sentence.replace("-", "")
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == "NOUN" or token.pos_ == "VERB") and (word != "left"):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list


def process_humanml3d(corpus):
    text_save_path = "./dataset/pose_data_raw/texts"
    desc_all = corpus
    for i in range(len(desc_all)):
        caption = desc_all.iloc[i]["caption"]
        start = desc_all.iloc[i]["from"]
        end = desc_all.iloc[i]["to"]
        name = desc_all.iloc[i]["new_joint_name"]
        word_list, pose_list = process_text(caption)
        tokens = " ".join(
            ["%s/%s" % (word_list[i], pose_list[i]) for i in range(len(word_list))]
        )
        with cs.open(pjoin(text_save_path, name.replace("npy", "txt")), "a+") as f:
            f.write("%s#%s#%s#%s\n" % (caption, tokens, start, end))
