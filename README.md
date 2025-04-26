# INF8225_Projet
# Étude Comparative des Algorithmes d'Apprentissage par Renforcement

Ce dépôt contient l'implémentation de différents algorithmes d'apprentissage par renforcement appliqués à plusieurs environnements de contrôle issus de la bibliothèque Gymnasium.

## Structure du projet

```
├── README.md
├── algorithms/
│   ├── DQN/
│   │   ├── DQN.py
│   │   └── replay_buffer.py
│   ├── RainbowDQN/
│   │   ├── DQNRainbow.py
│   │   ├── DQNRainbow-read_network.py
│   │   ├── replay_buffer.py
│   │   └── sum_tree.py
│   ├── DDPG/
│   │   ├── ddpg_lunar_launder_cartpole.py
│   │   └── replay_buffer.py
│   └── PPO/
│       └── ppo.py
└── results/
    ├── DQN/
    │   ├── DQN_CartPole-v1.png
    │   └── DQN_LunarLander-v3.png
    ├── RainbowDQN/
    │   ├── RainbowDQN_CartPole-v1.png
    │   └── RainbowDQN_LunarLander-v3.png
    ├── DDPG/
    │   ├── DDPG_CartPole.png
    │   ├── DDPG_CartPole_2.png
    │   ├── DDPG_LunarLaunder.png
    │   ├── DDPG_MountainCarContinuous.png
    │   └── DDPG_Pendulum.png
    └── PPO/
        ├── ppo_cartpole.png
        └── ppo_lunarlander.png
```

## Algorithmes implémentés

### Deep Q-Network (DQN)
DQN est un algorithme d'apprentissage par renforcement profond qui utilise un réseau neuronal pour approximer la fonction de valeur d'action Q. Il intègre des techniques clés comme l'experience replay et le réseau cible pour stabiliser l'apprentissage.

**Performances:**
- CartPole-v1: Score moyen de 110 après 15000 épisodes
- LunarLander-v3: Score moyen de 240 après 15000 épisodes

### Rainbow DQN
Rainbow DQN combine plusieurs améliorations du DQN standard, notamment le Double Q-learning, l'architecture Dueling, Prioritized Experience Replay, Multi-step Learning, Distributional RL, et Noisy Networks.

**Performances:**
- CartPole-v1: Score parfait de 500 après 2000 épisodes
- LunarLander-v3: Score moyen de 275 après 1500 épisodes

### Deep Deterministic Policy Gradient (DDPG)
DDPG est un algorithme actor-critic conçu pour les environnements à espace d'actions continu. Il utilise deux réseaux de neurones distincts: un acteur qui génère des actions et un critique qui évalue ces actions.

**Performances:**
- CartPole-v1: Score moyen de 480 après 3750 épisodes
- LunarLander-v3: Score moyen de -150 après 5000 épisodes
- Pendulum-v1: Score moyen de -800 après plusieurs milliers d'épisodes
- MountainCarContinuous-v0: Score moyen de -0.7 après 5000 épisodes

### Proximal Policy Optimization (PPO)
PPO est un algorithme d'optimisation de politique qui utilise un mécanisme de clipping pour limiter les mises à jour de la politique, ce qui le rend plus stable que d'autres méthodes de gradient de politique.

**Performances:**
- CartPole-v1: Score parfait de 500 après seulement 1600 épisodes
- LunarLander-v3: Score de 200 après 8200 épisodes

## Prérequis

- Python 3.7+
- PyTorch
- Gymnasium (OpenAI Gym)
- NumPy
- Matplotlib

Installation des dépendances:

```bash
pip install torch gymnasium numpy matplotlib
```

## Références

Ce projet s'inspire des implémentations suivantes:
- [Tutorials de reinforcement learning par tsmatz](https://github.com/tsmatz/reinforcement-learning-tutorials/tree/master)
- [Rainbow DQN en PyTorch par Lizhi-sjtu](https://github.com/Lizhi-sjtu/Rainbow-DQN-pytorch)

## Auteurs

- Hiroki Saï - Polytechnique Montréal
- Mamdouh Omar - Polytechnique Montréal
- Sala Glodi - Polytechnique Montréal
