# Improving Robotic Action Prediction with Reinforcement Learning

This project trains a dialogue manager that maps natural language queries to robotic actions. It combines supervised fine-tuning with Proximal Policy Optimization (PPO) to create a model that is more flexible.

---

## Overview

1. **SFT Baseline:**  
   A `Flan-T5` model is fine-tuned on a labeled dataset of `(query, action)` pairs to create an initial policy.

2. **RL Refinement:**  
   The baseline policy is then improved using PPO. A `LLaMA`-based Reward Model scores the policy's actions, providing a feedback for the RL loop. This allows the agent to learn from its mistakes and improve its precision over time.

---

## Usage

The project contains scripts for the main stages of the pipeline:

- `finetune_action/train_model.py`: Trains the initial baseline policy model.  
- `train_ppo.py`: Refines the fine-tuned model using the PPO loop with the LLaMA reward model.
