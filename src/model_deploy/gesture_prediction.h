#ifndef GESTURE_PREDICTION_H_
#define GESTURE_PREDICTION_H_

#include "accelerometer_handler.h"
#include "magic_wand_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

extern int predictGesture(float* output);
extern void prediction(int* angle, bool* flag);
extern void detect(int* angle, bool* flag);
extern void setupPrediction();
#endif  // G ESTURE_PREDICTION_H_