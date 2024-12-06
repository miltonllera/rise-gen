import os
import time
import argparse
import pprint
import pickle
import cma
import ray
import neptune
import torch as t
import torch.optim as optim
import numpy as np
from io import BytesIO
from PIL import Image
from glob import glob
from typing import List, Dict, Any
from tqdm import tqdm
from torch import nn
from tensorboardX import SummaryWriter
from neptune_tensorboard import enable_tensorboard_logging

from rl.algorithms import PPO
from model.vae.star_vae import StarVAE
from model.rl.robot import RobotEncoder, RobotActor, RobotCritic
from sim.env import RiseStochasticEnv
from utils.train_utils import RLTrainDirs, ExceptionCatcher, count_parameters
from scripts.common import build_robot


@ray.remote(num_gpus=0.9)
class RobotSimulationCollector:
    def __init__(
        self,
        rank: int,
        rl_dirs: RLTrainDirs,
        actor: RobotActor,
        critic: RobotCritic,
        min_actions: int,
        voxel_size: float,
    ):
        with ExceptionCatcher() as _:
            self.rank = rank
            self.rl_dirs = rl_dirs
            self.actor = actor.to("cuda:0")
            self.critic = critic.to("cuda:0")
            self.min_actions = min_actions
            self.voxel_size = voxel_size
            self.ppo = PPO(actor, critic, optim.AdamW, nn.MSELoss())

    def update_parameters(self, actor_state_dict, critic_state_dict):
        with ExceptionCatcher() as _:
            self.actor.load_state_dict(actor_state_dict)
            self.critic.load_state_dict(critic_state_dict)
            print(f"Collector {self.rank} parameters updated")
            return self.rank

    def collect(
        self,
        epoch,
        robot_ids: List[int],
        env_configs: List[str],
        robot_configs: List[str],
    ):
        with ExceptionCatcher() as _:
            if len(robot_configs) == 0:
                return []
            if len({len(robot_ids), len(env_configs), len(robot_configs)}) != 1:
                raise ValueError(
                    f"id, env config, robot config sizes "
                    f"[{len(robot_ids)}, {len(env_configs)}, {len(robot_configs)}] "
                    f"doesn't match"
                )
            env = RiseStochasticEnv(
                self.ppo,
                [0],
                env_configs,
                robot_configs,
                angles=[-1.4, -0.7, 0, 0.7, 1.4],
                debug_log_name=f"{self.rl_dirs.debug_log_path}"
                f"/debug_e_{epoch}_ra_{self.rank}.log",
            )
            result, observation, action, reward_state = env.run(epoch)

            episodes = []
            for robot_idx, success in tqdm(enumerate(result), total=len(result)):
                if success and len(observation[robot_idx]) > self.min_actions:
                    episode = []
                    episode_rewards = []
                    episode_costs = []
                    episode_contact_ratios = []
                    obs = observation[robot_idx]
                    act = action[robot_idx]
                    rst = reward_state[robot_idx]
                    last_a = None

                    start_com = rst[0]["com"]
                    end_com = rst[-1]["com"]

                    for i, (o, a, r) in enumerate(zip(obs, act, rst)):
                        com = r["com"]
                        reward_dir = com - start_com
                        reward_dir[2] = 0
                        reward_norm = np.linalg.norm(reward_dir)

                        if reward_norm > 0:
                            reward_dir /= reward_norm
                        else:
                            reward_dir = np.array([1.0, 0.0, 0.0])
                        voxel_num = len(r["voxel_positions"])
                        contact_ratio = (
                            np.sum(r["voxel_positions"][:, 2] < self.voxel_size * 5)
                            / voxel_num
                        )

                        reward = 0.0
                        if i < len(obs) - 1:
                            next_com = rst[i + 1]["com"]
                            next_movement = next_com - com
                            reward = np.dot(next_movement, reward_dir) / self.voxel_size
                            reward = max(reward, -0.1)
                            reward -= contact_ratio
                            if np.linalg.norm(next_movement) < 1e-3:
                                reward = -10.0

                        total = len(a[0])
                        changed = 0
                        if last_a is None:
                            for sub_a in a[0]:
                                if sub_a != 1:
                                    changed += 1
                        else:
                            for last_sub_a, sub_a in zip(last_a, a[0]):
                                if sub_a != last_sub_a:
                                    changed += 1
                        last_a = a[0]
                        action_cost = changed / total

                        episode_rewards.append(reward)
                        episode_costs.append(action_cost)
                        episode_contact_ratios.append(contact_ratio)
                        episode.append(
                            {
                                "state": o,
                                "action": {"action": a[0]},
                                "log_prob": a[1],
                                "reward": reward,
                                "terminal": i == len(obs) - 1,
                            }
                        )

                    movement = end_com - start_com
                    plane_distance = float(np.linalg.norm(movement[:2]))

                    mean_contact_ratio = np.mean(episode_contact_ratios)
                    episodes.append(
                        {
                            "id": robot_ids[robot_idx],
                            "reward": np.sum(episode_rewards),
                            "cost": np.sum(episode_costs),
                            "contact_ratio": np.mean(episode_contact_ratios),
                            "distance": plane_distance,
                            "rectified_distance": (
                                plane_distance / max(mean_contact_ratio, 1e-3)
                                if mean_contact_ratio < 0.25
                                else plane_distance
                            ),
                            "episode": episode,
                        }
                    )
                else:
                    episodes.append(
                        {
                            "id": robot_ids[robot_idx],
                            "reward": 0,
                            "cost": 0,
                            "contact_ratio": 0,
                            "distance": 0,
                            "rectified_distance": 0,
                            "episode": [],
                        }
                    )
            t.cuda.empty_cache()
            return episodes


@ray.remote(num_gpus=0.1)
class RobotEvolution:
    def __init__(
        self,
        source_files: List[str],
        env_vars: dict,
        generator_path: str,
        rl_dirs: RLTrainDirs,
        base_seed: int,
        epochs: int,
        env_config: str,
        voxel_size: float,
        max_torque: float,
        min_actions: int,
        collector_num: int,
        pop_size: int,
        rollout_num: int,
        rollout_epochs: int,
        actor_lr: float,
        critic_lr: float,
        ppo_batch_size: int,
        ppo_train_steps: int,
        ppo_accumulate_steps: int,
        ppo_save_interval: int,
        plot_robots: bool,
    ):
        with ExceptionCatcher() as _:
            for key, value in env_vars.items():
                os.environ[key] = value
            self.generator_path = generator_path
            self.rl_dirs = rl_dirs
            self.base_seed = base_seed
            self.epochs = epochs
            self.env_config = env_config
            self.voxel_size = voxel_size
            self.max_torque = max_torque
            self.min_actions = min_actions
            self.collector_num = collector_num
            self.pop_size = pop_size
            self.rollout_num = rollout_num
            self.rollout_epochs = rollout_epochs
            self.actor_lr = actor_lr
            self.critic_lr = critic_lr
            self.ppo_batch_size = ppo_batch_size
            self.ppo_train_steps = ppo_train_steps
            self.ppo_accumulate_steps = ppo_accumulate_steps
            self.ppo_save_interval = ppo_save_interval
            self.plot_robots = plot_robots

            ################################
            # Logging utilities
            ################################
            if "NEPTUNE_API_TOKEN" in os.environ and "NEPTUNE_PROJECT" in os.environ:
                mode = "async"
            else:
                mode = "offline"
                print("Note: Neptune is offline")

            self.neptune_run = neptune.init_run(mode=mode, source_files=source_files)
            self.neptune_run["general/root_path"] = rl_dirs.root_path
            self.neptune_run["general/log_path"] = rl_dirs.log_path
            self.neptune_run["general/debug_log_path"] = rl_dirs.debug_log_path
            self.neptune_run["general/results_path"] = rl_dirs.results_path
            self.neptune_run["general/records_path"] = rl_dirs.records_path
            self.neptune_run["general/ckpt_path"] = rl_dirs.ckpt_path
            enable_tensorboard_logging(self.neptune_run)
            self.writer = SummaryWriter(dirs.log_path, flush_secs=5)
            print("Neptune and Tensorboard Initialized")

            ################################
            # VAE setup
            ################################
            t.set_printoptions(threshold=10_000)
            t.manual_seed(base_seed)
            t.cuda.manual_seed(base_seed)
            np.random.seed(base_seed)
            self.vae = StarVAE.load_from_checkpoint(
                generator_path, map_location="cuda:0"
            )
            self.vae.eval()
            print("VAE loaded")

            if not os.path.exists("./data/vae_mean_var.data"):
                t.use_deterministic_algorithms(True)
                Xs = self.vae.get_samples(32)
                mu, logvar = self.vae.encode(Xs)
                latent = self.vae.rsample(mu, logvar)
                self.vae_mean = t.mean(latent, dim=0).detach().cpu()
                self.vae_var = t.var(latent, dim=0).detach().cpu()
                del latent
                del _
                t.save((self.vae_mean, self.vae_var), "./data/vae_mean_var.data")
                t.use_deterministic_algorithms(False)
            else:
                self.vae_mean, self.vae_var = t.load("./data/vae_mean_var.data")
            print("VAE mean and var loaded")

            ################################
            # RL setup
            ################################
            self.actor = RobotActor(
                RobotEncoder(
                    normalize_radius=voxel_size * self.vae.hparams.grid_size * 1.2 / 2,
                    grid_resolution=self.vae.hparams.grid_size,
                )
            )
            self.critic = RobotCritic(
                RobotEncoder(
                    normalize_radius=voxel_size * self.vae.hparams.grid_size * 1.2 / 2,
                    grid_resolution=self.vae.hparams.grid_size,
                )
            )
            actor_ref = ray.put(self.actor)
            critic_ref = ray.put(self.critic)
            self.actor = self.actor.to("cuda:0")
            self.critic = self.critic.to("cuda:0")
            print(f"Actor size: {count_parameters(self.actor)}")
            print(f"Critic size: {count_parameters(self.critic)}")
            self.ppo = PPO(
                self.actor,
                self.critic,
                optim.AdamW,
                nn.MSELoss(),
                batch_size=ppo_batch_size,
                actor_learning_rate=actor_lr,
                critic_learning_rate=critic_lr,
                gae_lambda=None,
                entropy_weight=0.01,
                discount=0.9,
            )
            print("RL initialized")
            ################################
            # Collector setup
            ################################
            self.collectors = [
                RobotSimulationCollector.remote(
                    c_idx, rl_dirs, actor_ref, critic_ref, min_actions, voxel_size
                )
                for c_idx in range(collector_num)
            ]
            print("Collectors created")

    def evolve(self):
        with ExceptionCatcher() as _:
            mean = self.vae_mean.numpy()
            std = np.sqrt(self.vae_var.numpy())
            morphology_es = cma.CMAEvolutionStrategy(
                x0=np.zeros_like(mean),
                sigma0=1,
                inopts={
                    "seed": self.base_seed,
                    "popsize": self.pop_size,
                    "CMA_stds": std,
                    "bounds": [
                        mean - 6 * std,
                        mean + 6 * std,
                    ],
                    "verbose": 9,
                },
            )
            epoch = 0
            top_k_latents = []
            k = 8
            robot_design_iters = 0
            while not morphology_es.stop():
                es_latents = morphology_es.ask()
                robot_design_epoch = epoch
                es_latents_with_memory = es_latents + top_k_latents
                robot_ids, robot_configs, robot_image_buffers, robot_scores = (
                    self._generate_robots(robot_design_epoch, es_latents_with_memory)
                )
                robot_image, robot_pil_image = self._generate_robot_images(
                    robot_image_buffers[:16]
                )
                self.writer.add_image("robot", robot_image, robot_design_epoch)
                self.neptune_run["run/robots"].append(robot_pil_image)
                self._save_design_result(
                    robot_design_epoch,
                    robot_design_iters,
                    es_latents_with_memory,
                    robot_image_buffers,
                )
                print(
                    f"Robots generated at epoch {robot_design_epoch} iter {robot_design_iters}"
                )
                # Invalid robots / robots with low performance has empty score list
                robot_sim_scores = [[] for _ in range(len(robot_configs))]

                # filter robot configs
                valid_robot_ids = []
                valid_robot_configs = []
                for idx, (id_, config, score) in enumerate(
                    zip(robot_ids, robot_configs, robot_scores)
                ):
                    if score > 0:
                        valid_robot_ids.append(id_)
                        valid_robot_configs.append(config)
                valid_robot_num = len(valid_robot_configs)
                valid_robot_ratio = valid_robot_num / len(robot_configs)
                print(
                    f"Valid robot configs: {valid_robot_num}, "
                    f"ratio: {valid_robot_ratio}"
                )
                self.writer.add_scalar(
                    "valid_robot_ratio", valid_robot_ratio, robot_design_iters
                )
                robot_design_iters += 1

                if valid_robot_ratio >= 0.5:
                    sim_ids = valid_robot_ids * self.rollout_num
                    sim_configs = valid_robot_configs * self.rollout_num

                    for rollout_epoch in range(self.rollout_epochs):
                        print(f"Start rollout at epoch {epoch}")

                        episodes = self._rollout(epoch, sim_ids, sim_configs)
                        for episode in episodes:
                            robot_sim_scores[episode["id"]].append(
                                episode["rectified_distance"]
                            )

                        if len(episodes) > 2:
                            self._train_ppo(epoch, episodes)
                            self._update_collector_parameters()
                        else:
                            print(
                                f"Too little valid episodes collected for training, skipping ppo"
                            )

                        epoch += 1
                else:
                    print("Low valid robot config ratio, skipping rollout")

                robot_report_scores = np.array(
                    [1e6 for _ in range(len(es_latents_with_memory))]
                )
                # CMA-ES will minimize scores
                for id_, scores in zip(valid_robot_ids, robot_sim_scores):
                    if len(scores) > 0:
                        # -Max seems fine too
                        robot_report_scores[id_] = -np.max(scores)
                    else:
                        # Some big value less that 1e6,
                        # encourage cma-es to generate valid outputs
                        # random is used to prevent cma-es stop
                        robot_report_scores[id_] = 1e3 + np.random.randn()

                top_k_indices = np.argsort(robot_report_scores)[:k]
                top_k_latents = [es_latents_with_memory[ki] for ki in top_k_indices]

                print(
                    f"Final scores of design "
                    f"epoch {robot_design_epoch} "
                    f"iter {robot_design_iters}: "
                    f"{robot_report_scores}"
                )
                self._save_design_scores(
                    robot_design_epoch, robot_design_iters, robot_report_scores
                )
                morphology_es.tell(es_latents, robot_report_scores[: len(es_latents)])
                morphology_es.logger.add()
                morphology_es.disp()

            self.neptune_run.stop()

    def _generate_robots(self, epoch: int, es_latents):
        with ExceptionCatcher() as _:
            t.use_deterministic_algorithms(True)
            latents = t.from_numpy(np.array(es_latents)).to(
                device="cuda:0", dtype=t.float32
            )
            logits = self.vae.generate_by_latent(latents)
            t.use_deterministic_algorithms(False)
            logits = logits.cpu()
            results = []
            robot_num = logits.shape[0]
            print(f"Submitting task for building {robot_num} robots")
            begin = time.time()
            for robot_id in range(logits.shape[0]):
                results.append(
                    build_robot.remote(
                        epoch,
                        robot_id,
                        logits[robot_id].numpy(),
                        self.rl_dirs,
                        self.max_torque,
                        self.plot_robots,
                        self.voxel_size,
                    )
                )
            print(f"Submission takes {time.time() - begin} s")

            print(f"Waiting for building {robot_num} robots")
            begin = time.time()
            results = ray.get(results)
            print(f"Building {robot_num} robots takes {time.time() - begin} s")

            robot_ids = list(range(len(results)))
            robot_configs = [r[0] for r in results]
            robot_image_buffers = [r[1] for r in results]
            robot_scores = [1 if len(c) > 0 else 0 for c in robot_configs]
            return robot_ids, robot_configs, robot_image_buffers, robot_scores

    @staticmethod
    def _generate_robot_images(image_buffers: List[BytesIO]):
        with ExceptionCatcher() as _:
            images = [Image.open(buf) for buf in image_buffers]
            widths, heights = zip(*(img.size for img in images))

            # Calculate the total width and height of the final stacked image
            total_width = max(widths)
            total_height = sum(heights)

            # Create a new blank image with the total dimensions
            stacked_image = Image.new("RGB", (total_width, total_height))

            # Paste each image into the new blank image
            y_offset = 0
            for img in images:
                stacked_image.paste(img, (0, y_offset))
                y_offset += img.size[1]

            return np.asarray(stacked_image).transpose((2, 0, 1)), stacked_image

    def _save_design_result(
        self,
        robot_design_epoch: int,
        robot_design_iters: int,
        es_latents: List[Any],
        robot_image_buffers: List[BytesIO],
    ):
        design_results = [
            (np.array(latent), image_buffer.read())
            for latent, image_buffer in zip(es_latents, robot_image_buffers)
        ]
        with open(
            f"{self.rl_dirs.log_path}/design_e_{robot_design_epoch}_it_{robot_design_iters}.data",
            "wb",
        ) as f:
            pickle.dump(design_results, f)

    def _save_design_scores(
        self,
        robot_design_epoch: int,
        robot_design_iters: int,
        robot_report_scores: np.ndarray,
    ):
        with open(
            f"{self.rl_dirs.log_path}/score_e_{robot_design_epoch}_it_{robot_design_iters}.data",
            "wb",
        ) as f:
            pickle.dump(robot_report_scores, f)

    def _rollout(self, epoch: int, sim_ids: List[int], sim_configs: List[str]):
        offset = 0
        results = []
        sim_per_collector = int(np.ceil(len(sim_ids) / self.collector_num))
        for collector in self.collectors:
            if offset >= len(sim_configs):
                break
            robot_ids = sim_ids[offset : offset + sim_per_collector]
            robot_configs = sim_configs[offset : offset + sim_per_collector]
            results.append(
                collector.collect.remote(
                    epoch,
                    robot_ids,
                    [self.env_config] * len(robot_configs),
                    robot_configs,
                )
            )
            offset += sim_per_collector
        performance = {
            "best_robot_metrics": {
                "rectified_distance": -np.inf,
                "collector_rank": None,
                "sim_index": None,
                "robot_id": None,
            },
            "collector_metrics": {},
        }

        episodes = []
        for rank, result in enumerate(results):
            sub_episodes = ray.get(result)
            if sub_episodes is None:
                print(
                    f"Exception caught in collector {rank}, returned none, " f"skipping"
                )
                continue
            performance["collector_metrics"][rank] = [
                {k: v for k, v in episode.items() if k != "episode"}
                for episode in sub_episodes
            ]
            for sim_index, episode in enumerate(sub_episodes):
                if (
                    len(episode["episode"]) > 70
                    and episode["rectified_distance"]
                    > performance["best_robot_metrics"]["rectified_distance"]
                ):
                    performance["best_robot_metrics"] = {
                        "rectified_distance": episode["rectified_distance"],
                        "collector_rank": rank,
                        "sim_index": sim_index,
                        "id": episode["id"],
                    }

            for episode in sub_episodes:
                if len(episode["episode"]) > 0:
                    episodes.append(episode)

        with open(
            f"{self.rl_dirs.debug_log_path}/debug_e_{epoch}_performance.log",
            "wb",
        ) as f:
            print(f"Performance data of epoch {epoch}")
            pprint.pprint(performance["best_robot_metrics"])
            pickle.dump(performance, f)
        print(f"Finish rollout at epoch {epoch}")
        return episodes

    def _train_ppo(self, epoch: int, episodes: List[Dict[str, Any]]):
        with ExceptionCatcher() as _:
            episode_lengths = []
            contact_ratios = []
            rewards = []
            costs = []
            distances = []
            rectified_distances = []
            per_step_rewards = []
            per_step_costs = []
            per_step_distances = []
            per_step_rectified_distances = []

            for episode in episodes:
                episode_lengths.append(len(episode["episode"]))
                contact_ratios.append(episode["contact_ratio"])
                rewards.append(episode["reward"])
                costs.append(episode["cost"])
                distances.append(episode["distance"])
                rectified_distances.append(episode["rectified_distance"])
                per_step_rewards.append(episode["reward"] / len(episode["episode"]))
                per_step_costs.append(episode["cost"] / len(episode["episode"]))
                per_step_distances.append(episode["distance"] / len(episode["episode"]))
                per_step_rectified_distances.append(
                    episode["rectified_distance"] / len(episode["episode"])
                )
                self.ppo.store_episode(episode["episode"], concatenate_samples=False)

            epoch_loss = 0
            for ppo_step in tqdm(
                range(self.ppo_train_steps), total=self.ppo_train_steps
            ):
                step_actor_loss = 0
                step_critic_loss = 0
                for optimizer in self.ppo.optimizers:
                    optimizer.zero_grad()
                for acc_step in range(self.ppo_accumulate_steps):
                    a_loss, c_loss = self.ppo.get_loss(concatenate_samples=False)
                    step_loss = (a_loss + c_loss) / self.ppo_accumulate_steps
                    step_loss.backward()
                    step_actor_loss += a_loss.item()
                    step_critic_loss += c_loss.item()
                    epoch_loss += step_loss.item()

                step_actor_loss /= self.ppo_accumulate_steps
                step_critic_loss /= self.ppo_accumulate_steps
                self.writer.add_scalar(
                    "step_actor_loss",
                    step_actor_loss,
                    epoch * ppo_train_steps + ppo_step,
                )
                self.writer.add_scalar(
                    "step_critic_loss",
                    step_critic_loss,
                    epoch * ppo_train_steps + ppo_step,
                )
                nn.utils.clip_grad_norm_(self.ppo.actor.parameters(), 4)
                nn.utils.clip_grad_norm_(self.ppo.critic.parameters(), 4)
                for optimizer in self.ppo.optimizers:
                    optimizer.step()
            epoch_loss /= ppo_train_steps * self.ppo_accumulate_steps
            self.writer.add_scalar("epoch_loss", epoch_loss, epoch)
            self.writer.add_scalar("max_episode_length", np.max(episode_lengths), epoch)
            self.writer.add_scalar(
                "mean_episode_length", np.mean(episode_lengths), epoch
            )
            self.writer.add_scalar("min_episode_length", np.min(episode_lengths), epoch)
            self.writer.add_scalar("max_contact_ratio", np.max(contact_ratios), epoch)
            self.writer.add_scalar("mean_contact_ratio", np.mean(contact_ratios), epoch)
            self.writer.add_scalar("min_contact_ratio", np.min(contact_ratios), epoch)
            self.writer.add_scalar(
                "max_per_step_reward", np.max(per_step_rewards), epoch
            )
            self.writer.add_scalar(
                "mean_per_step_reward", np.mean(per_step_rewards), epoch
            )
            self.writer.add_scalar(
                "min_per_step_reward", np.min(per_step_rewards), epoch
            )
            self.writer.add_scalar("max_per_step_cost", np.max(per_step_costs), epoch)
            self.writer.add_scalar("mean_per_step_cost", np.mean(per_step_costs), epoch)
            self.writer.add_scalar("min_per_step_cost", np.min(per_step_costs), epoch)
            self.writer.add_scalar(
                "max_per_step_distance", np.max(per_step_distances), epoch
            )
            self.writer.add_scalar(
                "mean_per_step_distance", np.mean(per_step_distances), epoch
            )
            self.writer.add_scalar(
                "min_per_step_distance", np.min(per_step_distances), epoch
            )
            self.writer.add_scalar(
                "max_per_step_rectified_distance",
                np.max(per_step_rectified_distances),
                epoch,
            )
            self.writer.add_scalar(
                "mean_per_step_rectified_distance",
                np.mean(per_step_rectified_distances),
                epoch,
            )
            self.writer.add_scalar(
                "min_per_step_rectified_distance",
                np.min(per_step_rectified_distances),
                epoch,
            )
            self.writer.add_scalar("max_reward", np.max(rewards), epoch)
            self.writer.add_scalar("mean_reward", np.mean(rewards), epoch)
            self.writer.add_scalar("min_reward", np.min(rewards), epoch)
            self.writer.add_scalar("max_cost", np.max(costs), epoch)
            self.writer.add_scalar("mean_cost", np.mean(costs), epoch)
            self.writer.add_scalar("min_cost", np.min(costs), epoch)
            self.writer.add_scalar("max_distance", np.max(distances), epoch)
            self.writer.add_scalar("mean_distance", np.mean(distances), epoch)
            self.writer.add_scalar("min_distance", np.min(distances), epoch)
            self.writer.add_scalar(
                "max_rectified_distance", np.max(rectified_distances), epoch
            )
            self.writer.add_scalar(
                "mean_rectified_distance", np.mean(rectified_distances), epoch
            )
            self.writer.add_scalar(
                "min_rectified_distance", np.min(rectified_distances), epoch
            )
            self.writer.add_histogram("reward", np.array(rewards), epoch)
            self.writer.add_histogram("cost", np.array(costs), epoch)
            self.writer.add_histogram(
                "reward_cost ratio", np.array(rewards) / (np.array(costs) + 1e-3), epoch
            )
            self.writer.add_histogram("distance", np.array(distances), epoch)

            self.ppo.finish_update()

            np.save(
                os.path.join(dirs.log_path, f"epoch={epoch}-contact-ratio.npy"),
                np.array(contact_ratios),
            )
            np.save(
                os.path.join(dirs.log_path, f"epoch={epoch}-distance.npy"),
                np.array(distances),
            )
            np.save(
                os.path.join(dirs.log_path, f"epoch={epoch}-rectified-distance.npy"),
                np.array(rectified_distances),
            )
            np.save(
                os.path.join(dirs.log_path, f"epoch={epoch}-reward.npy"),
                np.array(rewards),
            )
            np.save(
                os.path.join(dirs.log_path, f"epoch={epoch}-cost.npy"), np.array(costs)
            )
            np.save(
                os.path.join(dirs.log_path, f"epoch={epoch}-reward-cost-ratio.npy"),
                np.array(rewards) / (np.array(costs) + 1e-3),
            )

            t.cuda.empty_cache()
            if epoch % self.ppo_save_interval == 0:
                t.save(
                    {
                        "actor": self._cpu_state_dict(self.actor),
                        "critic": self._cpu_state_dict(self.critic),
                    },
                    os.path.join(
                        dirs.ckpt_path,
                        f"epoch={epoch}-rectified_distance={np.mean(rectified_distances)}.ckpt",
                    ),
                )

    def _update_collector_parameters(self):
        actor_state_dict = ray.put(self._cpu_state_dict(self.actor))
        critic_state_dict = ray.put(self._cpu_state_dict(self.critic))
        ranks = ray.get(
            [
                collector.update_parameters.remote(actor_state_dict, critic_state_dict)
                for collector in self.collectors
            ]
        )
        del actor_state_dict
        del critic_state_dict
        print(f"Finish update parameters for collectors: {ranks}")

    @staticmethod
    def _cpu_state_dict(model):
        return {k: v.cpu() for k, v in model.state_dict().items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_path", type=str, default="./data/ckpt/vae.ckpt")
    parser.add_argument("--env_config_path", type=str, default="./data/env.rsc")
    parser.add_argument("--root_path", default="./data/rl-result")
    parser.add_argument("--log_sub_path", default="logs")
    parser.add_argument("--debug_log_sub_path", default="debug-logs")
    parser.add_argument("--ckpt_sub_path", default="ckpt")
    parser.add_argument("--results_sub_path", default="results")
    parser.add_argument("--records_sub_path", default="records")
    parser.add_argument("--base_seed", default=42)
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--voxel_size", default=0.01)
    parser.add_argument("--max_torque", default=6)
    parser.add_argument("--min_actions", default=30)
    parser.add_argument("--collector_num", default=4)
    parser.add_argument("--pop_size", default=64)
    parser.add_argument("--rollout_num", default=2)
    parser.add_argument("--rollout_epochs", default=20)

    parser.add_argument("--actor_lr", default=6e-5)
    parser.add_argument("--critic_lr", default=6e-5)
    parser.add_argument("--ppo_batch_size", default=16)
    parser.add_argument("--ppo_train_steps", default=60)
    parser.add_argument("--ppo_accumulate_steps", default=4)
    parser.add_argument("--ppo_save_interval", default=10)
    parser.add_argument("--plot_robots", default=True)

    parser.add_argument("--num_cpus", default=50)
    parser.add_argument("--num_collectors", default=2)

    args = parser.parse_args()

    ray.init(num_cpus=50)
    print("Ray initialized")
    with open(args.env_config_path, "r") as f:
        env_config = f.read()

    dirs = RLTrainDirs(
        root_path=args.root_path,
        log_sub_path=args.log_sub_path,
        debug_log_sub_path=args.debug_log_sub_path,
        ckpt_sub_path=args.ckpt_sub_path,
        results_sub_path=args.results_sub_path,
        records_sub_path=args.records_sub_path,
    )
    env_vars = dict(os.environ)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    code_root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    evo = RobotEvolution.remote(
        source_files=[
            f
            for f in glob(f"{code_root_dir}/**/*.py", recursive=True)
            if not f.startswith(f"{code_root_dir}/venv")
        ]
        + [f"{code_root_dir}/data/env.rsc"],
        env_vars=env_vars,
        generator_path=args.generator_path,
        rl_dirs=dirs,
        base_seed=args.base_seed,
        epochs=args.epochs,
        env_config=env_config,
        voxel_size=args.voxel_size,
        max_torque=args.max_torque,
        min_actions=args.min_actions,
        collector_num=args.collector_num,
        pop_size=args.pop_size,
        rollout_num=args.rollout_num,
        rollout_epochs=args.rollout_epochs,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        ppo_batch_size=args.ppo_batch_size,
        ppo_train_steps=args.ppo_train_steps,
        ppo_accumulate_steps=args.ppo_accumulate_steps,
        ppo_save_interval=args.ppo_save_interval,
        plot_robots=args.plot_robots,
    )
    ray.get(evo.evolve.remote())
    time.sleep(5)
    ray.shutdown()
