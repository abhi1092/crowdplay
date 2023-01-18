# Offline Algorithms

## Run Offline algo

To run the program, change the following:

``python3 main.py --seed=123 --task=space_invaders_left --algo=BC --gpu=False``

### Parameters

* **algo**: Default is BC. Other options are: BCQ, CQL, IQN, DQN, SAC
* **task**: Default is space_invaders_left. Other options are:
  space_invaders_right, space_invaders_insideout, space_invaders_outsidein,
  space_invaders_rowbyrow, space_invaders_mturk, riverraid_left,
  riverraid_right, riverraid_mturk, qbert_mturk, beam_rider_mturk
* **gpu**:  Default is False. Set it to True if gpu is availble.

## Additional scripts

`offline-atari-get-human-performance.py` can be used to get the baseline human performance for each task, useful for computing relative performance.

`offline-rollout.py` and `ai-rollout.py` are highly experimental scripts used to roll out offline and online RL agents, and to store trajectory data into a crowdplay dataset. `offline-rollout.py` was used to generate the t-SNE plots comparing human and BC agent behavior in the ICLR paper. `ai-rollout.py` could be used to roll out AI agents usually used for multiagent environments in a CrowdPlay deployment, but without any human players present.



### RDCQL experiment
```shell
PYTHONPATH=`pwd` python offline/offline-atari-rdcql.py --algo RDCQL --track --wandb_api_key 1c56954c3534056d7a0734857f6f991fd31925a3 --discriminator_kl_penalty_coef 1.0 --discriminator_clip_ratio 0.5 --discriminator_weight_temp 1.0 --discriminator_lr 1e-4 --gpu 0 --task=riverraid_left
PYTHONPATH=`pwd` python offline/offline-atari-rdcql.py --algo RDCQL --track --wandb_api_key 1c56954c3534056d7a0734857f6f991fd31925a3 --discriminator_kl_penalty_coef 5.0 --discriminator_clip_ratio 2.5 --discriminator_weight_temp 0.5 --discriminator_lr 1e-4 --gpu 1

```
