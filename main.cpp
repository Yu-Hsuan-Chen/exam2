#include "mbed.h"
#include "uLCD_4DGL.h"
#include "mbed_rpc.h"
#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"
#include "accelerometer_handler.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <vector>
#include <math.h>
 

#define PI 3.14159265
#define label_num 3
struct Config {

  // This must be the same as seq_length in the src/model_train/config.py
  const int seq_length = 64;

  // The number of expected consecutive inferences for each gesture type.
  const int consecutiveInferenceThresholds[label_num] = {20, 10, 10};

  const char* output_message[label_num] = {
        "UP", "Z", "Circle"};
};
Config config;

struct Data {
    int16_t XYZ[100][3];
    int index = 0;
}


// MQTT/////////////////////////////////////////////
WiFiInterface *wifi;
volatile int message_num = 0;
volatile int arrivedcount = 0;
bool closed = false;
const char* topic = "Mbed";
Thread mqtt_thread(osPriorityAboveNormal);
Thread mqtt_send_thread(osPriorityHigh);
EventQueue mqtt_send_queue;
MQTT::Client<MQTTNetwork, Countdown> *client;
///////////////////////////////////////////////////

//Mode 1////////////////////////////////////////////
bool flag1;
Thread gesture_thread(osPriorityNormal);
void gestureMode(Arguments *in, Reply *out);
RPCFunction rpcgesture(&gestureMode, "gestureMode");
EventQueue queue(32 * EVENTS_EVENT_SIZE);
int indexR = 0;
Data collected_data[10];
int gesture_data[10];
/////////////////////////////////////////////////////

//Mode 2/////////////////////////////////////////////
int flag2;
int threshold;
double theta;
int16_t DataXYZ[3] = {0};
EventQueue detect_queue;
Thread detect_thread(osPriorityNormal);
void detectionMode(Arguments *in, Reply *out);
RPCFunction rpcdetect(&detectionMode, "detectionMode");
//////////////////////////////////////////////////////

//Mode 3//////////////////////////////////////////////
bool close_RPC = false;
void closeAll(Arguments *in, Reply *out);
RPCFunction rpcclosed(&closeAll, "closeAll");
//////////////////////////////////////////////////////


//LEDs
DigitalOut myled1(LED1); //mode1
DigitalOut myled2(LED2); //mode2
DigitalOut myled3(LED3); //initialization
InterruptIn btn(USER_BUTTON);
BufferedSerial pc(USBTX, USBRX);
int mode = 1;
//uLCD////////////////////////////////////////////////
uLCD_4DGL uLCD(D1, D0, D2);
Thread uLCD_thread(osPriorityNormal);


int angles[5] = {30, 35, 40, 45, 50};
int idx;

//prediction////////////////////////////////////////
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
bool should_clear_buffer = false;
bool got_data = false;
tflite::ErrorReporter* error_reporter;
TfLiteTensor* model_input;
tflite::MicroInterpreter* interpreter;
int input_length;
int gesture_index;
/////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////////////////

void messageArrived(MQTT::MessageData& md) {
    MQTT::Message &message = md.message;
    char msg[300];
    sprintf(msg, "Message arrived: QoS%d, retained %d, dup %d, packetID %d\r\n", message.qos, message.retained, message.dup, message.id);
    printf(msg);
    ThisThread::sleep_for(1000ms);
    char payload[300];
    sprintf(payload, "Payload %.*s\r\n", message.payloadlen, (char*)message.payload);
    printf(payload);
    ++arrivedcount;
}

void publish_message(int* mode) {
    message_num++;
    MQTT::Message message;
    char buff[100];
    sprintf(buff, "Num : %d, Gesture: %d", message_num, gesture_index);
    message.qos = MQTT::QOS0;
    message.retained = false;
    message.dup = false;
    message.payload = (void*) buff;
    message.payloadlen = strlen(buff) + 1;
    int rc = client->publish(topic, message);

    printf("rc:  %d\r\n", rc);
    printf("Puslish message: %s\r\n", buff);
}

void receive_message() {
    while (true) {
            client->yield(500);
            ThisThread::sleep_for(500ms);
    }
}

void connectMQTT() {
    wifi = WiFiInterface::get_default_instance();
    if (!wifi) {
            printf("ERROR: No WiFiInterface found.\r\n");
            // return -1;
    }
    printf("\nConnecting to %s...\r\n", MBED_CONF_APP_WIFI_SSID);
    int ret = wifi->connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);
    if (ret != 0) {
            printf("\nConnection error: %d\r\n", ret);
            // return -1;
    }
    NetworkInterface* net = wifi;
    MQTTNetwork mqttNetwork(net);
    client = new MQTT::Client<MQTTNetwork, Countdown> (mqttNetwork);
    //TODO: revise host to your IP
    const char* host = "192.168.160.34";
    printf("Connecting to TCP network...\r\n");

    SocketAddress sockAddr;
    sockAddr.set_ip_address(host);
    sockAddr.set_port(1883);

    printf("address is %s/%d\r\n", (sockAddr.get_ip_address() ? sockAddr.get_ip_address() : "None"),  (sockAddr.get_port() ? sockAddr.get_port() : 0) ); //check setting

    int rc = mqttNetwork.connect(sockAddr);//(host, 1883);
    if (rc != 0) {
            printf("Connection error.");
            // return -1;
    }
    printf("MQTT successfully connected...\r\n");

    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
    data.MQTTVersion = 3;
    data.clientID.cstring = "Mbed";

    if ((rc = client->connect(data)) != 0){
            printf("Fail to connect MQTT\r\n");
    }
    if (client->subscribe(topic, MQTT::QOS0, messageArrived) != 0){
            printf("Fail to subscribe\r\n");
    }
    mqtt_send_thread.start(callback(&mqtt_send_queue, &EventQueue::dispatch_forever));
    // btn.rise(mqtt_send_queue.event(&publish_message, &mode));
    int num = 0;
    while (num != 3) {
            client->yield(100);
            ++num;
    }
    while (!closed) {
            // if (closed) break;
            client->yield(500);
            ThisThread::sleep_for(500ms);
    }
    
    printf("Ready to close MQTT Network......\n");

    // if ((rc = client->unsubscribe(topic)) != 0) {
    //         printf("Failed: rc from unsubscribe was %d\n", rc);
    // }
    // if ((rc = client->disconnect()) != 0) {
    // printf("Failed: rc from disconnect was %d\n", rc);
    // }
    delete client;
    mqttNetwork.disconnect();
    close_RPC = true;
    printf("Successfully closed!\n");
}

// void LCD() {
//     uLCD.media_init(); // initialize uSD card
//     if(mode == 1) {
//         uLCD.cls();
//         uLCD.locate(0, 0);
//         uLCD.color(LGREY);
//         uLCD.printf("\n Angle list:\n");
//         uLCD.printf("\n  30 35 40 45 50\n");
//         uLCD.color(0x02C874);
//         uLCD.printf("\n Selected Angle\n\n");
//         uLCD.text_width(2); 
//         uLCD.text_height(2);
//         uLCD.locate(0, 4);
//         uLCD.printf("   %d \n", angle);
//         ThisThread::sleep_for(2s);        
//     }
//     while(mode == 1) {
//         uLCD.text_width(2); 
//         uLCD.text_height(2);
//         uLCD.locate(0, 4);
//         uLCD.printf("   %d \n", angle);       
//     }
//     ///////////////////////////////////
//     uLCD.cls();
//     ThisThread::sleep_for(1s);
//     if(mode == 2) {
//         uLCD.text_width(1); 
//         uLCD.text_height(1);
//         uLCD.locate(0, 0);
//         uLCD.color(LGREY);
//         uLCD.printf("\n Threshold Angle: \n");
//         uLCD.text_width(2); 
//         uLCD.text_height(2);
//         uLCD.printf("   %d \n", threshold);
//         uLCD.color(0x02C874);
//         uLCD.text_width(1); 
//         uLCD.text_height(1);
//         uLCD.locate(0, 5);
//         uLCD.printf("\n Tilt Angle :\n\n");    
//         uLCD.text_width(2); 
//         uLCD.text_height(2);
//         uLCD.locate(0, 4);
//         uLCD.printf("  %.3f \n", theta);
//         uLCD.text_width(1); 
//         uLCD.text_height(1);
//         uLCD.locate(0, 10);
//         uLCD.color(0x4EFEB3);
//         uLCD.printf("\n(Updating \n every 2 sec... ) \n");        
//     }
//     while(mode == 2) {
//         uLCD.text_width(2); 
//         uLCD.text_height(2);
//         uLCD.locate(0, 4);
//         uLCD.printf("  %.3f \n", theta);
//     }
//     if(mode != 1 && mode != 2) {
//         uLCD.text_width(2); 
//         uLCD.text_height(2);
//         uLCD.color(RED);
//         uLCD.printf("\n EROOR \n"); 
//     }
// }

int main() {
    BSP_ACCELERO_Init();
    mqtt_thread.start(connectMQTT);
    char buf[256], outbuf[256];
    FILE *devin = fdopen(&pc, "r");
    FILE *devout = fdopen(&pc, "w");
    //RPC loop
    while(!close_RPC) {
        memset(buf, 0, 256);
        for (int i = 0; ; i++) {
            char recv = fgetc(devin);
            if (recv == '\n') {
                printf("\r\n");
                break;
            }
            buf[i] = fputc(recv, devout);
        }
        //Call the static call method on the RPC class
        RPC::call(buf, outbuf);
        printf("%s\r\n", outbuf);
   }
   mqtt_send_thread.join();
   mqtt_thread.join();
//    uLCD_thread.join();
}


//Mode 1///////////////////////////


void gestureUI() {
    prediction(&flag1);
    myled1.write(0);
    printf("Close Mode 1 !\n\n");
}

void gestureMode(Arguments *in, Reply *out) {
    printf("Enter Mode 1!\n\n");
    mode = 1;
    // uLCD_thread.start(LCD);
    myled1.write(1);
    gesture_thread.start(gestureUI);
}

int predictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

void detect(bool* flag) {
  idx = 0;
  while(*flag) {
    // startRecord();
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,input_length, should_clear_buffer);
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }
    // Analyze the results to obtain a prediction
    gesture_index = predictGesture(interpreter->output(0)->data.f);
    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;
    // Produce an output
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
    }    
    // stopRecord();    
  }
}

void setupPrediction() {
// Set up logging.
  BSP_ACCELERO_Init();
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    // return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                                tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                                tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                                tflite::ops::micro::Register_RESHAPE(), 1);

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    // return -1;
  }

  input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    // return -1;
  }

  error_reporter->Report("Prediction set up successful...\n");
}

void prediction(bool*flag) {
  (*flag) = true;
  setupPrediction();
  detect(flag);
  return;
}


void startRecord(void) {
    BSP_ACCELERO_AccGetXYZ(pDataXYZ);
    data[indexR++].XYZ = pDataXYZ;
    gesture_data[indexR++] = gesture_index;
    indexR = indexR % 10;
}

void stopRecord(void) {
   for (auto &i : idR)
      queue.cancel(i);
}



//Mode 2///////////////////////////

// double calculateTheta(double a1, double a2, double a3, double b1, double b2, double b3) {
//     double len_x = sqrt(a1 * a1 + a2 * a2 + a3 * a3);
//     double len_y = sqrt(b1 * b1 + b2 * b2 + b3 * b3);
//     int dot = a1 * b1 + a2 * b2 + a3 * b3;
//     double cosin = dot / (len_x * len_y);
//     double theta = acos(cosin) * 180.0 / PI;
//     return theta;
// }

// //Detect Function
// void detecting(int* threshold) {
//     printf("\nPlace your mbed on table...\n");
//     printf("Wait until the led3 light...\n");
//     ThisThread::sleep_for(5s);
//     BSP_ACCELERO_AccGetXYZ(DataXYZ);
//     double b1 = DataXYZ[0];
//     double b2 = DataXYZ[1];
//     double b3 = DataXYZ[2];
//     myled3.write(1);
//     printf("\nStart to detect the tilt angle!\n");
//     printf("The threshold angle: %d\n", *threshold);
//     int num = 0;
//     ThisThread::sleep_for(2s);
//     while(flag2 < 10) {
//         BSP_ACCELERO_AccGetXYZ(DataXYZ);
//         theta = calculateTheta(DataXYZ[0], DataXYZ[1], DataXYZ[2], b1, b2, b3);
//         printf("\nNum: %d\nNow tilt angle: %lf\n\n", num++, theta);
//         if(theta >= (*threshold)) {
//             flag2++;
//             publish_message(&mode);
//         }
//         ThisThread::sleep_for(2s);
//     }
// }
// //Thread Function
// void detectAngle(int* threshold) {
//     detecting(threshold);
//     myled2.write(0);
//     myled3.write(0);
//     printf("\nClose Mode 2 !\n");    
// }
// //RPC Function
// void detectionMode(Arguments *in, Reply *out) {
//     printf("Enter Mode 2!\n");
//     flag2 = 0;
//     myled2.write(1);
//     threshold = in->getArg<double>();
//     mode = 2;
//     detect_thread.start(callback(&detectAngle, &threshold));
//     while(1) {
//         if(flag2 == 10) {
//             break;
//         } 
//         ThisThread::sleep_for(500ms);
//     }
//     ThisThread::sleep_for(2s);
//     detect_thread.join();
//     printf("\nUse \" /closeAll/run \" to close the program !\n");
// }

// //Closed/////////////////////////////
// void closeAll(Arguments *in, Reply *out) {
//     closed = true;
// }


    