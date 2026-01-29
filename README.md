# The Self-Evidencing Reasoning Engine (SERE)

**A Non-Connectionist Architecture for Generative Reasoning and Planning**

This repository contains the reference implementation and ongoing research for the Self-Evidencing Reasoning Engine (SERE), a cognitive architecture grounded in the **Free Energy Principle (FEP)** and **Active Inference**. It represents a fundamental departure from the connectionist paradigm of Deep Neural Networks (DNNs) and Large Language Models (LLMs).

## 1. Vision: Beyond Curve-Fitting to Self-Evidencing

The prevailing AI orthodoxy of reward maximization and loss minimization on static datasets has created powerful pattern-matching systems, but not truly intelligent agents. Such systems lack transparency, require vast quantities of data, and do not possess a robust, internal model of their world.

The SERE is predicated on a different normative principle: **self-evidencing**. An intelligent agent, like a biological organism, must actively maintain its structural and functional integrity against the entropic tendencies of its environment. It achieves this by continuously minimizing **Variational Free Energy (VFE)**—an information-theoretic bound on surprise. In short, the agent's primary imperative is to make the world understandable by constantly updating its internal generative model and acting to fulfill its own predictions.

This project aims to build such an agent for the domain of Natural Language, treating text understanding not as classification, but as a continuous cycle of belief updating, reasoning, and information-seeking.

## 2. Core Architectural Pillars

The SERE is a "non-neural" architecture that replaces opaque weight matrices with transparent, factorized tensor representations and local, biologically plausible learning rules.

| Pillar | Technology | Role & Purpose | Implemented In |
| :--- | :--- | :--- | :--- |
| **State Representation**| Factorized POMDPs | Models the world with explicit, disentangled hidden states (e.g., Semantic vs. Syntactic). | `models/generative_model.py` |
| **Scalability Engine**| Tensor Train (TT) Decompositions | Compresses the exponentially large belief states and transition dynamics into a tractable, linear-scaling format. Solves the "curse of dimensionality." | `models/tensor_train.py` |
| **Learning Mechanism**| Riemannian Manifold Optimization | Enables online, gradient-based learning directly on the low-rank TT-manifold, avoiding backpropagation and preserving computational efficiency. | *(Theoretical Framework)* |
| **Dynamic Transitions**| Selective State Space Models (Mamba) | Implements context-dependent dynamics, allowing the agent to selectively attend to information, akin to attention but with linear complexity. | *(Theoretical Framework)* |
| **Objective Function**| Variational Free Energy (VFE) | The universal objective for perception (inference) and learning, balancing model accuracy with complexity. | `models/free_energy.py` |
| **Action Selection**| Expected Free Energy (EFE) | The mechanism for planning and reasoning. The agent chooses actions that minimize future surprise, balancing goal-seeking (**pragmatic value**) with curiosity (**epistemic value**). | `models/policies.py` |

## 3. How It Works: The "Think-Act" Loop

The SERE operates in a continuous loop of perception, planning, and action, driven entirely by the free energy imperative.

1.  **Perception as Inference**: When presented with a new observation (a token), the agent's belief state (a Tensor Train) is updated to best explain the new evidence. This is not a feedforward pass; it is a variational message-passing routine that finds the posterior beliefs that minimize VFE.

2.  **Planning as Inference**: The agent considers potential future action sequences (policies). For each policy, it calculates the **Expected Free Energy (EFE)**. This is the core of the reasoning engine. Policies that resolve uncertainty (high epistemic value) are prioritized when the agent is "confused," while policies that achieve preferred outcomes (high pragmatic value) are chosen when the agent is confident.

3.  **Action & Learning**: The agent selects an action by sampling from the distribution over policies. After acting, the model parameters (the TT-cores of the generative model) are updated via Riemannian optimization to account for the prediction error from the previous step.

## 4. Project Status & Usage

This repository contains a foundational implementation of the SERE architecture.

### Current Implementation
-   **Core Logic**: The `DiscretePOMDP`, VFE/EFE calculations, and belief-updating mechanisms are fully implemented in PyTorch.
-   **Factorization**: The model correctly uses a factorized semantic/syntactic state space.
-   **Proof-of-Concept**: The `train_shakespeare.py` script demonstrates the model's ability to learn grammatical structure from raw text using a dense (non-Tensor-Train) implementation.

### Running the Code

**1. Validate the Core Logic:**
To understand the mathematical mechanics in a simple, controlled environment, run the validation script:
```bash
python hai_text/scripts/validate_pomdp.py
```

**2. Train the Foundational Model:**
To train the dense prototype on the Tiny Shakespeare dataset:
```bash
python hai_text/scripts/train_shakespeare.py
```
This script will download the data and begin training the 512x512 state-space model, saving checkpoints to the `hai_text/data/` directory.

## 5. Discrepancy Analysis & Future Work

This project is an active area of research. The complete vision outlined in the theoretical documentation is the goal, and the current codebase is the first major milestone. For a detailed analysis of the differences between the current code and the full research plan, please refer to the project's internal documentation. The next phases of development will focus on:

1.  **Tensorization**: Replacing the dense `A` and `B` tensors in `DiscretePOMDP` with the `TTMatrix` and `TTTensor` classes from `tensor_train.py`.
2.  **Manifold Optimization**: Implementing a Riemannian optimizer in JAX to replace the current `torch.optim.Adam`.
3.  **Hierarchical Scaling**: Using the `HierarchicalPOMDP` class to build and train deep temporal models.

## License

This project is licensed under the MIT License.
