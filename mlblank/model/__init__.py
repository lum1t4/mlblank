from mlblank.core.context import ModelRegistry, load_model

from mlblank.model.segformer import SegformerConfig, SegformerForSemanticSegmentation
from mlblank.model.tracknet import TrackNet, TrackNetConfig
from mlblank.model.yolov8 import YOLOv8Config


ModelRegistry.register_model('nvidia/segformer-b0', SegformerForSemanticSegmentation, SegformerConfig(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], decoder_hidden_size=256, num_labels=1), "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/model.safetensors")
ModelRegistry.register_model('nvidia/segformer-b1', SegformerForSemanticSegmentation, SegformerConfig(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=256, num_labels=1), "https://huggingface.co/nvidia/segformer-b1-finetuned-ade-512-512/resolve/main/pytorch_model.bin")
ModelRegistry.register_model('nvidia/segformer-b2', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))
ModelRegistry.register_model('nvidia/segformer-b3', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 4, 18, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))
ModelRegistry.register_model('nvidia/segformer-b4', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 8, 27, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))
ModelRegistry.register_model('nvidia/segformer-b5', SegformerForSemanticSegmentation, SegformerConfig(depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, num_labels=1))

ModelRegistry.register_model('ultralytics/yolov8n', YOLOv8Config(num_channels=3, num_labels=80, width=0.25, ratio=2.0, depth=0.33, inplace=True),"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt")
ModelRegistry.register_model('ultralytics/yolov8s', YOLOv8Config(num_channels=3, num_labels=80, width=0.50, ratio=2.0, depth=0.33, inplace=True),"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt")
ModelRegistry.register_model('ultralytics/yolov8m', YOLOv8Config(num_channels=3, num_labels=80, width=0.75, ratio=1.5, depth=0.67, inplace=True),"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt")
ModelRegistry.register_model('ultralytics/yolov8l', YOLOv8Config(num_channels=3, num_labels=80, width=1.00, ratio=1.0, depth=1.00, inplace=True),"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt")
ModelRegistry.register_model('ultralytics/yolov8x', YOLOv8Config(num_channels=3, num_labels=80, width=1.25, ratio=1.0, depth=1.00, inplace=True),"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt")

ModelRegistry.register_model('NCTU/TrackNet', TrackNet, TrackNetConfig(num_channels=3, num_labels=1))


__all__ = ['load_model']
