/*
 * TensorFlow Lite Micro Inference Runner — STM32F4xx
 * Runs the INT8 anomaly detection model on accelerometer data.
 */
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "anomaly_model_int8.h"   /* converted model header */
#include "main.h"
#include <string.h>

#define TENSOR_ARENA_SIZE  (32 * 1024)
#define SEQ_LEN            64

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static const tflite::Model*         model_ptr;
static tflite::MicroInterpreter*    interpreter;
static TfLiteTensor*                input_tensor;
static TfLiteTensor*                output_tensor;

void inference_init(void) {
    model_ptr = tflite::GetModel(anomaly_model_int8_tflite);
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interp(
        model_ptr, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter   = &static_interp;
    interpreter->AllocateTensors();
    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);
}

/*
 * run_inference: accepts SEQ_LEN float samples, returns 1=anomaly, 0=normal
 */
int8_t run_inference(float* samples) {
    /* Quantize input: x_q = (x / scale) + zero_point */
    float scale      = input_tensor->params.scale;
    int32_t zp       = input_tensor->params.zero_point;
    for (int i = 0; i < SEQ_LEN; i++) {
        input_tensor->data.int8[i] = (int8_t)((samples[i] / scale) + zp);
    }

    interpreter->Invoke();

    float out_scale = output_tensor->params.scale;
    int32_t out_zp  = output_tensor->params.zero_point;
    float probability = (output_tensor->data.int8[0] - out_zp) * out_scale;

    return (probability > 0.5f) ? 1 : 0;  /* 1 = anomaly */
}
