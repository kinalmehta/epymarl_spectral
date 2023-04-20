# epymarl_spectral

This is the official implementation of "Effects of Spectral Normalization in Multi-agent Reinforcement Learning" accepted at IJCNN-2023

MARL algorithms with Spectral Normalization
EPyMARL-Spectral is an extension of [EPyMARL](https://github.com/uoe-agents/epymarl), and includes:
- Spectral Normalization and Spectral Regularization support in neural-networks (actor, critic or value-function)
- Updated logging to be more structured and use `torch.utils.tensorboard.SummaryWriter` and [wand.ai](https://wandb.ai/)

# Installation Instructions
- Create a new conda environment and activate it
  ```bash
  conda create -n spectral_marl python=3.9
  conda activate spectral_marl
  ```
- Install smac
  ```bash
  bash install_sc2.sh
  ```
- Install other required packages
  ```python
  pip install -r requirements.txt
  ```

# Run an experiment
- Train with spectral normalization applied to last layer of the critic
  ```python
  python src_spectral/main.py --config=mappo --env-config=sc2_sparse with env_args.map_name=27m_vs_30m \
          standardise_returns=True use_rnn=True policy_spectral="nnnn" critic_spectral="nny" t_max=40050000
  ```
- Train with no spectral normalization
  ```python
  python src_spectral/main.py --config=mappo --env-config=sc2_sparse with env_args.map_name=27m_vs_30m \
          standardise_returns=True use_rnn=True policy_spectral="nnnn" critic_spectral="nnn" t_max=40050000
  ```
- The parameters `critic_spectral` and `policy_spectral` are used to control whether to apply spectral normalization to the models

# Citation
If you use this code in your project, please cite the following paper:
```bibtex
@article{mehta2022effects,
      title={Effects of Spectral Normalization in Multi-agent Reinforcement Learning}, 
      author={Kinal Mehta and Anuj Mahajan and Pawan Kumar},
      year={2022},
      eprint={2212.05331},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2212.05331},
}
```

