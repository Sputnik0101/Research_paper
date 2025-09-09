# Space SER (RTX 3050 Edition)

Emotion & tone recognition for space HRI, optimized for **RTX 3050 4GB**.

## Features
- Space-comms augmentation: band-limit (300–3400 Hz), 8 kHz hop, noise, burst packet loss
- Wav2Vec2 fine-tuning **memory-optimized** (fp16, gradient checkpointing, partial unfreeze)
- Baseline-ready: frozen encoder + linear head
- Evaluation (macro-F1, confusion matrix) + simple inference script
- Optional ROS2 node for behavior adaptation

## Setup
```bash
conda create -n space-ser python=3.10 -y
conda activate space-ser
pip install -r requirements.txt
```

> If you use CUDA, install matching `torch`/`torchaudio` wheels if needed.

## Data
Put audio WAVs in `./data`:
- RAVDESS: `./data/ravdess/...`
- (optional) IEMOCAP: `./data/iemocap/...`
- (optional) Apollo comms clips: `./data/apollo/...`

Build metadata:
```bash
python src/prepare_data.py
```

## Train (RTX 3050 friendly)
```bash
python src/train_ser.py
```

## Evaluate & Plot
```bash
python src/evaluate_ser.py
```

## Inference (single file)
```bash
python src/infer.py --wav path/to/sample.wav
```

## Tips for 4GB GPU
- Keep `batch_size=1`, use `gradient_accumulation_steps` to simulate larger batch
- Start with `freeze_encoder=true`, then let script unfreeze the last `N` layers automatically
- If you still hit OOM, set `unfreeze_last_n_layers: 0` (pure linear probe) and/or set `fp16: false` for CPU runs

## ROS2 (optional)
Linux only. Source your ROS2 workspace, then:
```bash
python src/ros2_emotion_node.py
```
Drop `.wav` files into `data/inbox/` and watch logs for actions.
