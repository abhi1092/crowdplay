import argparse
import d3rlpy
import os, sys
import uuid

from d3rlpy.algos import DiscreteCQL
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

from d3rlpy_algos.rd_discrete_cql import RDDiscreteCQL
from d3rlpy_algos.wandb_logger import WandbLoggerWrapper

ALGOS = {
    "CQL": DiscreteCQL,
    "RDCQL": RDDiscreteCQL,
}

def main(args):
    dataset, env = get_atari(args.dataset)

    experiment_name = f"{args.algo}_{args.dataset}_{args.seed}_{uuid.uuid4()}"
    d3rlpy.seed(args.seed)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    baseclass = ALGOS[args.algo]
    alg_cls = baseclass

    if args.track:
        import wandb
        if args.wandb_api_key:
          wandb.login(key=args.wandb_api_key)

        configs = {**vars(args)}
        configs["logdir"] = os.path.join(args.logdir, experiment_name)

        # Check if the specified configuration has been run
        prefixed_configs = {f"config.{key_of_interest}": configs[key_of_interest] for key_of_interest in [
                  "dataset",
                  "algo",
                  "seed",
                  "discriminator_clip_ratio",
                  "discriminator_kl_penalty_coef",
                  "discriminator_weight_temp",
                  "discriminator_lr"]}
        api = wandb.Api()
        old_runs = list(api.runs(path="improbableai_zwh/drr_cql_atari",
                filters={
                   "state": "finished",
                    **prefixed_configs,
                }))

        if len(old_runs) > 0:
          print("Run exists")
          sys.exit()

        run = wandb.init(
            name=experiment_name,
            project="drr_cql_atari",
            entity="improbableai_zwh",
            tags=args.tags,
            config=configs,
            # sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            settings=wandb.Settings(start_method='fork')
        )

        class WrappedAlgorithm(WandbLoggerWrapper, baseclass):
            pass

        alg_cls = WrappedAlgorithm

    cql = alg_cls(
        n_frames=4,  # frame stacking
        q_func_factory=args.q_func,
        scaler='pixel',
        use_gpu=args.gpu,

        discriminator_clip_ratio=args.discriminator_clip_ratio,
        discriminator_kl_penalty_coef=args.discriminator_kl_penalty_coef,
        discriminator_weight_temp=args.discriminator_weight_temp,
        discriminator_lr=args.discriminator_lr,
        )

    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=100,
            scorers={
                'environment': evaluate_on_environment(env, epsilon=0.05),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer
            },
            experiment_name=experiment_name,
            logdir=args.logdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='breakout-mixed-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--q-func',
                        type=str,
                        default='mean',
                        choices=['mean', 'qr', 'iqn', 'fqf'])
    parser.add_argument('--algo',
                        type=str,
                        default='CQL',
                        choices=['CQL', 'RDCQL'])
    parser.add_argument('--sampler',
                        type=str,
                        default='uniform',
                        choices=['uniform'])
    parser.add_argument("--discriminator_clip_ratio", type=float, default=5.0)
    parser.add_argument("--discriminator_weight_temp", type=float, default=1.0)
    parser.add_argument("--discriminator_lr", type=float, default=1e-4)
    parser.add_argument("--discriminator_kl_penalty_coef", type=float, default=0.01)
    parser.add_argument('--gpu', type=int)
    parser.add_argument("--logdir", type=str, default="/tmp")
    parser.add_argument('--track', action="store_true", default=False)
    parser.add_argument('--tags', nargs="+", type=str, default=[])
    parser.add_argument('--wandb_api_key', type=str, default=None)
    args = parser.parse_args()
    main(args)