# Edge AI Embedded Inference

[![CI](https://github.com/Qandel-Embedded/edge-ai-embedded-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/Qandel-Embedded/edge-ai-embedded-inference/actions)
[![TFLite](https://img.shields.io/badge/TensorFlow%20Lite-INT8-orange)](training/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ML anomaly detection on microcontrollers: Python training pipeline exports INT8 TFLite model → C inference runner on STM32F4.

## Pipeline
```
Raw sensor data → CNN model training (Python/TF) → INT8 quantization → .tflite → STM32 C inference
```

## Quick Start (Training)
```bash
pip install -r requirements.txt
python training/train_anomaly_model.py
# Outputs: anomaly_model_int8.tflite (~18 KB)
```

## Deploy to STM32
1. Run `xxd -i anomaly_model_int8.tflite > anomaly_model_int8.h`
2. Add TFLM library to STM32CubeIDE project
3. Include `firmware/inference_runner.c`
4. Call `inference_init()` once, then `run_inference(samples)` per reading

## Performance (STM32F411 @ 100MHz)
| Metric | Value |
|--------|-------|
| Model size | ~18 KB flash |
| RAM usage | < 32 KB |
| Inference time | ~42 ms |
| Accuracy | 94.8% |

---
**Portfolio:** https://ahmedqandel.com | Available for hire on Upwork
