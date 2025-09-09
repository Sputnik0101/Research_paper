# Minimal ROS2 node reacting to predicted emotion labels.
import rclpy
from rclpy.node import Node
import torch, torchaudio, yaml, time, os, glob
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

RESPONSES = {
  "angry": "Acknowledged. Switching to support mode and notifying ground.",
  "fearful": "I am here. Initiating safety checklist and contacting mission control.",
  "sad": "Consider a short break. I will keep monitoring.",
  "happy": "Great! Proceeding with nominal operations.",
  "neutral": "Continuing as planned.",
  "calm": "Copy that. Maintaining current plan.",
  "disgust": "Noted. I will re-plan around the issue.",
  "surprised": "Registered anomaly. Marking event for analysis."
}

class EmotionNode(Node):
    def __init__(self):
        super().__init__('emotion_node')
        cfg = yaml.safe_load(open("config.yaml"))
        self.sr = cfg["sample_rate"]
        ckpt = "./outputs/checkpoints"
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(ckpt)
        self.feat  = Wav2Vec2FeatureExtractor.from_pretrained(ckpt, sampling_rate=self.sr)
        self.get_logger().info("Emotion node ready.")

    def classify(self, wav):
        if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
        inputs = self.feat([wav.squeeze(0).numpy()], sampling_rate=self.sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = logits.argmax(-1).item()
        label = self.model.config.id2label[pred]
        return label

def main(args=None):
    rclpy.init(args=args)
    node = EmotionNode()
    inbox = "./data/inbox"
    os.makedirs(inbox, exist_ok=True)
    node.get_logger().info(f"Watching {inbox} for .wav files...")
    try:
        seen = set()
        while rclpy.ok():
            for f in glob.glob(f"{inbox}/*.wav"):
                if f in seen: 
                    continue
                wav, sr = torchaudio.load(f)
                if sr != node.sr:
                    wav = torchaudio.functional.resample(wav, sr, node.sr)
                label = node.classify(wav)
                msg = RESPONSES.get(label, RESPONSES["neutral"])
                node.get_logger().info(f"{os.path.basename(f)} -> {label} | action: {msg}")
                seen.add(f)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
