import argparse
import os

import imageio
import numpy as np

from research.utils.config import Config
from research.utils.trainer import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default=None, required=False, help="Path to save the gif")
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--num-ep", type=int, default=1, help="Number of episodes")
    parser.add_argument("--every-n-frames", type=int, default=2, help="Save every n frames to the gif.")
    parser.add_argument("--width", type=int, default=160, help="Width of image")
    parser.add_argument("--height", type=int, default=120, help="Height of image")
    parser.add_argument("--strict", action="store_true", default=False, help="Strict")
    parser.add_argument(
        "--end-on-success", action="store_True", default=False, help="Terminate gif on success condition."
    )
    parser.add_argument(
        "--override",
        metavar="KEY=VALUE",
        nargs="+",
        default=[],
        help="Set kv pairs used as args for the entry point script.",
    )
    parser.add_argument("--max-len", type=int, default=1000, help="maximum length of an episode.")
    args = parser.parse_args()

    assert args.path.endswith(".pt"), "Must provide a model checkpoint"
    config = Config.load(os.path.dirname(args.path))
    config["checkpoint"] = None  # Set checkpoint to None

    # Overrides
    print(args.override)

    # Overrides
    for override in args.override:
        items = override.split("=")
        key, value = items[0].strip(), "=".join(items[1:])
        # Progress down the config path (seperated by '.') until we reach the final value to override.
        config_path = key.split(".")
        config_dict = config
        while len(config_path) > 1:
            config_dict = config_dict[config_path[0]]
            config_path.pop(0)
        config_dict[config_path[0]] = value

    if len(args.override) > 0:
        print(config)

    save_gif = args.output is not None
    render_kwargs = dict(mode="rgb_array", width=args.width, height=args.height) if save_gif else dict()

    model = load(config, args.path, device=args.device, strict=args.strict)
    model.eval_mode()  # Place the model in evaluation mode
    env = model.env if model.env is not None else model.eval_env
    ep_rewards, ep_lengths = [], []
    frames = []
    for ep in range(args.num_ep):
        obs = env.reset()
        done = False
        ep_reward, ep_length = 0, 0
        frame = env.render(**render_kwargs)
        if save_gif:
            frames.append(frame)
        while not done and ep_length <= args.max_len:
            batch = dict(obs=obs)
            if hasattr(env, "_max_episode_steps"):
                batch["horizon"] = env._max_episode_steps - ep_length
            action = model.predict(batch)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            if ep_length % args.every_n_frames == 0:
                frame = env.render(**render_kwargs)
                if save_gif:
                    frames.append(frame)
            if args.end_on_success and (info.get("success", False) or info.get("is_success", False)):
                print("[research] Episode success, terminating early.")
                done = True
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)
        print("Finished Episode. Reward:", ep_reward, "Length:", ep_length)

    print("Overall. Reward:", np.mean(ep_rewards), "Length:", np.mean(ep_lengths))
    if save_gif:
        # Cut the frames
        print("Saving a gif of", len(frames), "Frames")
        imageio.mimsave(args.output, frames[:: args.every_n_frames])
