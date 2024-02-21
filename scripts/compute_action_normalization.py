import argparse

import numpy as np

import research
from research.utils.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config.")
    parser.add_argument("--clip", type=float, default=4.0, help="std-dev clipping for min-max normalization")
    args = parser.parse_args()

    config = Config.load(args.config)
    config = config.parse()
    dataset_class = None if config["dataset"] is None else vars(research.datasets)[config["dataset"]]
    dataset_kwargs = config["dataset_kwargs"]
    assert issubclass(
        dataset_class, research.datasets.ReplayBuffer
    ), "Must use replay buffer for normalization computation"
    dataset_kwargs["distributed"] = False  # Ensure that we load all of the data.
    observation_space, action_space = config.get_spaces()

    # Exclude all observations for faster loading
    exclude_keys = list(dataset_kwargs.get("exclude_keys", []))
    exclude_keys.extend(["obs.*", "reward", "discount"])  # Cannot remove done!
    dataset_kwargs["exclude_keys"] = exclude_keys

    # Create the dataset, exclude everything but actions so we don't load it
    dataset = dataset_class(observation_space, action_space, **dataset_kwargs)

    # Loop through the dataset to get all the actions, ignoring the dummy ones.
    all_actions = []
    storage = dataset._storage
    # NOTE: we add one to starts for the offset :)
    starts, ends = storage.starts + 1, storage.ends
    for start, end in zip(starts, ends):
        actions = storage["action"][start:end]
        all_actions.append(actions)

    all_actions = np.concatenate(all_actions, axis=0)

    # Compute all normalization possibilities.

    action_min, action_max = np.min(all_actions, axis=0), np.max(all_actions, axis=0)

    print("Low: ", action_min)
    print("High: ", action_max)

    action_mean, action_std = np.mean(all_actions, axis=0), np.std(all_actions, axis=0)

    print("Mean: ", action_mean)
    print("Std: ", action_std)

    # Compute the min / max after clipping.
    gaussian_normalized_actions = (all_actions - action_mean) / action_std
    # now clip everything
    gaussian_normalized_actions = np.clip(gaussian_normalized_actions, a_min=-args.clip, a_max=args.clip)
    # now re-normalize everthing
    clipped_low, clipped_high = np.min(gaussian_normalized_actions, axis=0), np.max(gaussian_normalized_actions, axis=0)

    print("Clipped Low: ", clipped_low)
    print("Clipped High: ", clipped_high)
