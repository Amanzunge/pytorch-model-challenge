# Pytorch Model Challenge

## Project Overview

This project implements a multi-stream event modeling system for employee-related data. Each event stream (EventType, Location, CostCode, JobCode, etc.) is modeled independently and then fused using cross-attention to predict multiple target fields related to employee events.

The training pipeline is fully containerized using Docker and supports both model training and Tensorboard visualization through `docker-compose`.

---

## Approach and Modeling Intuition

**Architecture Summary:**
- Multiple input streams are provided, each corresponding to a different aspect of an employee event.
- Each stream is passed through its own independent Transformer encoder.
- The encoded representations are then fused together using a Cross-Attention mechanism.
- The fused vector is decoded into multiple outputs, each corresponding to a different target field (e.g., EventType, ActorRecordId, LocationId, etc.).
- A separate decoder head is built for each target field.
- Loss is calculated independently for each field and averaged to form the final training loss.
- Each stream is processed independently, allowing easy addition or removal of streams in the future.
- Different event types often have different statistical properties. Separate Transformer encoders allow the model to learn stream-specific features without interference.
- Instead of naively concatenating stream outputs, cross-attention enables a flexible, learnable combination of multiple streams.
- Using separate decoders allows the model to specialize for each output field without conflict.
- The architecture is extensible to more input streams or more output fields without major redesign.

**Training Data:**
- Since real-world datasets were not provided, synthetic data is generated dynamically.
- The synthetic data is generated in a structured way to allow the model to learn meaningful patterns (not purely random).

**Loss Calculation:**
- For each output field, CrossEntropyLoss is calculated separately.
- The total loss is the mean of all individual field losses, ensuring balanced learning across outputs.

---

## On Completion Questions

**1. What is the meaning of the outputs from the encoder streams?**

Each encoder stream outputs a contextualized embedding for its respective input sequence. These embeddings capture the temporal and semantic structure of each event stream separately (e.g., how EventType evolves over time for an employee). The outputs are then used for cross-attention fusion, contributing to a global understanding of the employee event context.

**2. What are some improvements you may make to this model?**

- Introduce positional encoding explicitly into each stream.
- Implement a learnable query for cross-attention instead of static zero initialization.
- Add residual connections and normalization after fusion.
- Increase model depth and number of heads for better capacity.
- Fine-tune synthetic data generation for more realistic patterns.

**3. How would you conduct a beam search using this model? How would the model need to change?**

Beam search could be used during inference if the model were adapted to autoregressive decoding (predicting next event token step-by-step). The model would need to:
- Output sequential predictions (next event step) conditioned on previous outputs.
- Maintain multiple candidate sequences (beams) during decoding.
- Expand and prune beams based on cumulative probabilities.
Currently, the model is not autoregressive, so decoder design would need to change to output one token at a time.

**4. Why would you conduct a beam search?**

Beam search increases the chance of finding better output sequences compared to greedy decoding. It explores multiple high-probability paths instead of just the single most probable one. This would be useful if the model were generating sequences of events or employee actions rather than single labels.

**5. How would you convert this model's decoder layer into a diffusion model?**

To convert into a diffusion model:
- The decoder would predict noise-added versions of target variables at different diffusion steps.
- Training would involve learning to denoise from noisy intermediate states back to clean targets.
- A time-step embedding would be added to the input of the decoders.
- Loss would be computed between predicted denoised outputs and original targets at each step.

**6. How would this model behave differently if this is a diffusion model?**

- Instead of direct classification, the model would gradually denoise and reconstruct target variables.
- Training dynamics would shift from straightforward cross-entropy minimization to denoising score matching.
- Inference would be slower (multiple diffusion steps needed).
- Likely better sample diversity, but more computational overhead.

---

## Setup Instructions

### Install Python dependencies manually

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To run training manually:

```bash
python -m modeling.train_model
```

### Running with Docker

Make sure Docker Desktop is installed. Then:

```bash
docker compose up --build
```

- This will build two containers:
  - One for training (`model_training`)
  - One for Tensorboard (`model_tensorboard`)
- Both containers share the `/runs` directory for logging.

Tensorboard will be available at:

```
http://localhost:6006/
```

### Tensorboard Details

- All training and test losses are logged.
- Loss per target field is logged separately.
- Time per batch step is logged.

---

## Folder Structure

```
/
├── data/
│   └── events_loader.py
├── modeling/
│   ├── models/
│   │   └── model.py
│   └── train_model.py
├── Training.Dockerfile
├── Tensorboard.Dockerfile
├── docker-compose.yml
├── requirements.txt
├── training_config.json
├── README.md
├── INTUITION.md
├── ModelArchitecture.png
└── .gitignore
```

---

## Notes

- The project has been tested locally with Python 3.10 and Torch.
- Training and Tensorboard work both locally and inside Docker containers.
- Synthetic data was used but structured to allow meaningful loss learning.
- Model architecture matches the assignment intuition fully.

---

