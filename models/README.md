# models/

Download the YOLOX-s person detector here before running notebook 07:

```bash
curl -L -o models/yolox_s.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
```

The notebooks save the trained models in this folder:

| File | Made by | What it is |
|---|---|---|
| `helmet_color_model.keras` | notebook 04 | Model 1: photo of a helmet → 5 color probabilities |
| `helmet_and_color.keras` | notebook 06 | Model 2 + Model 1 joined: head crop → P(helmet) and color |
| `helmet_person_head.keras` | notebook 08 | Output A': ResNet50 vector of a whole person → P(helmet) |
| `final_system.keras` | notebook 10 | Final system: person + head → P(helmet) and color |

Keypoint R-CNN (notebooks 09 and 10) and the ResNet50 ImageNet weights download by themselves the first time.
