import os
import json
import numpy as np
import torch 

class Logger:
    def __init__(self, use_wandb=False, project="rl-policy-gradient", name=None, log_dir="logs"):
        self.use_wandb = use_wandb
        self.step = 0

        os.makedirs(log_dir, exist_ok=True)
        self.file = open(os.path.join(log_dir, "log.jsonl"), "a")

        if use_wandb:
            import wandb
            wandb.init(project=project, name=name)
            self.wandb = wandb

    def log(self, data: dict):
        data["step"] = self.step

        # print
        print(data)

        # save file
        self.file.write(json.dumps(data) + "\n")
        self.file.flush()

        # wandb
        if self.use_wandb:
            self.wandb.log(data, step=self.step)

    def log_video(self, frames, fps=30):
        if not self.use_wandb:
            return

        video = np.array(frames)           # (T, H, W, C)
        video = video.transpose(0, 3, 1, 2)  # (T, C, H, W)

        self.wandb.log({
            "video": self.wandb.Video(video, fps=fps, format="mp4")
        }, step=self.step)

    def set_step(self, step):
        self.step = step

    def close(self):
        self.file.close()
        if self.use_wandb:
            self.wandb.finish()

def record_video(env, policy, device, max_steps=500):
    frames = []
    obs, _ = env.reset()

    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

        with torch.no_grad():
            action, _, _ = policy.predict(obs_tensor, deterministic_bool = True)

        obs, _, terminated, truncated, _ = env.step(action.cpu().numpy())

        if terminated or truncated:
            break

    return frames

