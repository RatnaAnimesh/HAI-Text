# Hierarchical Active Inference (HAI) for Text Generation

A PyTorch implementation of **Discrete Hierarchical Active Inference (HAI)** models for structured text generation. This framework demonstrates that **structured, factorized priors** (Semantic $\times$ Syntactic) can achieve superior sample efficiency and interpretability compared to unstructured monolithic neural networks (LLMs).

## \U0001f9e0 Key Features

*   **Discrete POMDP Architecture**: Exact mathematical implementation of Partially Observable Markov Decision Processes with discrete latent states.
*   **Factorized Latent Space**: Disentangles **Semantic** (meaning) and **Syntactic** (grammar) factors ($S_{sem} \times S_{syn}$), enabling combinatorial generalization.
*   **Hierarchical Coupling**: Implements top-down modulation where high-level beliefs constrain low-level transition dynamics.
*   **Active Inference**: Supports goal-directed behavior via **Expected Free Energy (EFE)** minimization.
*   **Differentiable Learning**: Trained end-to-end on raw text (e.g., Shakespeare) by minimizing **Variational Free Energy (VFE)** via SGD.

## \U0001f4ca Results

Using a compact model with **512 Semantic** $\times$ **512 Syntactic** states (approx. 262k latent capacity):

*   **Sample Efficiency**: Converges to robust grammar in <2 epochs on the Tiny Shakespeare dataset.
*   **Biological Plausibility**: State space capacity matches the functional dimensionality of a cortical macro-column.
*   **Jargon-Free Generation**: Learns to produce coherent English word structures and punctuation without the billion-parameter scale of Transformers.

## \U0001f680 Quick Start

### Prerequisites
*   Python 3.8+
*   PyTorch
*   NumPy

### Installation
```bash
git clone https://github.com/yourusername/hai-text.git
cd hai-text
pip install torch numpy
```

### Training (Shakespeare)
To train the model on the Tiny Shakespeare dataset from scratch:

```bash
python hai_text/scripts/train_shakespeare.py
```

This will:
1.  Download the dataset.
2.  Initialize a 512x512 Discrete POMDP.
3.  Train for 2 epochs (approx. 20 hours on CPU/M3 Air).
4.  Save checkpoints to `data/model_v2_512.pt`.
5.  Generate sample text demonstrating learned structure.

### Validation (Toy Task)
To verify the core mathematical correctness on a simple "Subject-Verb-Object" task:

```bash
python hai_text/scripts/validate_pomdp.py
```

## \U0001f4c2 Project Structure

*   `hai_text/models/`
    *   `generative_model.py`: The core `DiscretePOMDP` class (learnable `nn.Module`).
    *   `inference.py`: Fixed-point Variational Message Passing (Mean Field) implementation.
    *   `free_energy.py`: Computation of VFE (Complexity + Accuracy) and EFE.
    *   `hierarchical.py`: Two-level coupling logic.
*   `hai_text/training/`: Dataset loaders and SGD training loops.
*   `hai_text/scripts/`: Experiments and validation scripts.

## \U0001f4da Citation

If you use this code in your research, please cite:

```bibtex
@article{YourName2026HAI,
  title={Hierarchical Active Inference for Data-Efficient Text Generation},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

## License
MIT
