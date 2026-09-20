#!/usr/bin/env python3
"""
export_mobile_onnx.py
Export 3 checkpoints in checkpointyolov8n to optimized mobile ONNX formats:
1. QualityGate (Light Package, thumbnail 64x64) -> models_mobile/quality_gate.onnx (~46 KB)
2. MQTone (Tone-mapping corrector, 64x64)      -> models_mobile/mqtone.onnx (~125 KB)
3. YOLOv8n (Dual-task detector, 640x640)         -> models_mobile/yolov8n_cashvision.onnx (~12.8 MB)
"""

import os
import sys
from pathlib import Path
import torch

# Fix UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import quality_gate
import mqtone
import train_c2
from ultralytics import YOLO

def export_all():
    out_dir = Path("models_mobile")
    out_dir.mkdir(exist_ok=True)
    print(f"=== EXPORTING MODELS TO {out_dir} ===")

    # 1. Export QualityGate (Light Package)
    print("\n[1/3] Exporting QualityGate (Lightweight Sensory Gate)...")
    qg_weights = Path("checkpointyolov8n/quality_gate_weights.pt")
    if not qg_weights.exists():
        raise FileNotFoundError(f"Not found: {qg_weights}")
    
    qg = quality_gate.QualityGate(thumbnail_size=64)
    qg.load_state_dict(torch.load(str(qg_weights), map_location="cpu"))
    qg.eval()

    qg_onnx_path = out_dir / "quality_gate.onnx"
    dummy_qg = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        qg,
        dummy_qg,
        str(qg_onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=18
    )
    print(f"-> Exported: {qg_onnx_path} ({os.path.getsize(qg_onnx_path) / 1024:.1f} KB)")

    # 2. Export MQTone (Illumination Correction)
    print("\n[2/3] Exporting MQTone (Photometric Tone-Mapping Network)...")
    mq_weights = Path("checkpointyolov8n/icnet_weights.pt")
    if not mq_weights.exists():
        raise FileNotFoundError(f"Not found: {mq_weights}")
    
    mq = mqtone.MQTone(thumbnail_size=64, grid_size=8)
    mq.load_state_dict(torch.load(str(mq_weights), map_location="cpu"))
    mq.eval()

    mq_onnx_path = out_dir / "mqtone.onnx"
    dummy_mq = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        mq,
        dummy_mq,
        str(mq_onnx_path),
        input_names=["input"],
        output_names=["corrected_image"],
        opset_version=18
    )
    print(f"-> Exported: {mq_onnx_path} ({os.path.getsize(mq_onnx_path) / 1024:.1f} KB)")

    # 3. Export YOLOv8n (Banknote & Tear Detector)
    print("\n[3/3] Extracting and exporting YOLOv8n Detector...")
    full_pipe_weights = Path("checkpointyolov8n/full_pipeline_weights.pt")
    if not full_pipe_weights.exists():
        raise FileNotFoundError(f"Not found: {full_pipe_weights}")
    
    # Load pipeline and extract detector backbone
    pipe = train_c2.C2DetectionPipeline(weights_path="yolov8n.pt", correction_method="mqtone")
    pipe.load_state_dict(torch.load(str(full_pipe_weights), map_location="cpu"))
    det_state = {k.replace("detector.", ""): v for k, v in pipe.state_dict().items() if k.startswith("detector.")}

    # Create intermediate checkpoint for Ultralytics
    temp_pt = out_dir / "temp_yolov8n_cashvision.pt"
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.model.load_state_dict(det_state)
    torch.save({"model": yolo_model.model.half()}, str(temp_pt))

    # Export ONNX via Ultralytics exporter (NMS and output formatting)
    yolo_loaded = YOLO(str(temp_pt))
    exported_file = yolo_loaded.export(format="onnx", imgsz=640, half=False)
    
    target_yolo_onnx = out_dir / "yolov8n_cashvision.onnx"
    if Path(exported_file) != target_yolo_onnx:
        if target_yolo_onnx.exists():
            target_yolo_onnx.unlink()
        Path(exported_file).rename(target_yolo_onnx)

    if temp_pt.exists():
        temp_pt.unlink()

    print(f"-> Exported: {target_yolo_onnx} ({os.path.getsize(target_yolo_onnx) / (1024*1024):.2f} MB)")

    print("\n=======================================================")
    print("SUCCESS! Created 3 production ONNX models ready for mobile deployment:")
    print(f" 1. {qg_onnx_path} (Light Package)")
    print(f" 2. {mq_onnx_path} (Tone Corrector)")
    print(f" 3. {target_yolo_onnx} (YOLOv8n Detector)")
    print("=======================================================")

if __name__ == "__main__":
    export_all()
