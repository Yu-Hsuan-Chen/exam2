#include "mbed.h"
#include "uLCD_4DGL.h"
#include "mbed_rpc.h"
#include "gesture_prediction.h"
#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"
#include <vector>
#include <math.h>
 

#define PI 3.14159265

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
int angle;
bool flag1;
Thread gesture_thread(osPriorityNormal);
void gestureMode(Arguments *in, Reply *out);
RPCFunction rpcgesture(&gestureMode, "gestureMode");
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
    MQTT::Message message;
    char buff[100];

    if((*mode) == 1) {
        flag1 = false;
        gesture_thread.join();
        sprintf(buff, "%d", angle);        
    }else {  //Mode 2
        sprintf(buff, "The tilt angle: %.3f is larger than threshold angle(%d)!", theta, threshold);  
    }
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
    btn.rise(mqtt_send_queue.event(&publish_message, &mode));
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

void LCD() {
    uLCD.media_init(); // initialize uSD card
    if(mode == 1) {
        uLCD.cls();
        uLCD.locate(0, 0);
        uLCD.color(LGREY);
        uLCD.printf("\n Angle list:\n");
        uLCD.printf("\n  30 35 40 45 50\n");
        uLCD.color(0x02C874);
        uLCD.printf("\n Selected Angle\n\n");
        uLCD.text_width(2); 
        uLCD.text_height(2);
        uLCD.locate(0, 4);
        uLCD.printf("   %d \n", angle);
        ThisThread::sleep_for(2s);        
    }
    while(mode == 1) {
        uLCD.text_width(2); 
        uLCD.text_height(2);
        uLCD.locate(0, 4);
        uLCD.printf("   %d \n", angle);       
    }
    ///////////////////////////////////
    uLCD.cls();
    ThisThread::sleep_for(1s);
    if(mode == 2) {
        uLCD.text_width(1); 
        uLCD.text_height(1);
        uLCD.locate(0, 0);
        uLCD.color(LGREY);
        uLCD.printf("\n Threshold Angle: \n");
        uLCD.text_width(2); 
        uLCD.text_height(2);
        uLCD.printf("   %d \n", threshold);
        uLCD.color(0x02C874);
        uLCD.text_width(1); 
        uLCD.text_height(1);
        uLCD.locate(0, 5);
        uLCD.printf("\n Tilt Angle :\n\n");    
        uLCD.text_width(2); 
        uLCD.text_height(2);
        uLCD.locate(0, 4);
        uLCD.printf("  %.3f \n", theta);
        uLCD.text_width(1); 
        uLCD.text_height(1);
        uLCD.locate(0, 10);
        uLCD.color(0x4EFEB3);
        uLCD.printf("\n(Updating \n every 2 sec... ) \n");        
    }
    while(mode == 2) {
        uLCD.text_width(2); 
        uLCD.text_height(2);
        uLCD.locate(0, 4);
        uLCD.printf("  %.3f \n", theta);
    }
    if(mode != 1 && mode != 2) {
        uLCD.text_width(2); 
        uLCD.text_height(2);
        uLCD.color(RED);
        uLCD.printf("\n EROOR \n"); 
    }
}

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
   uLCD_thread.join();
}


//Mode 1///////////////////////////

//Thread Function
void gestureUI() {
    prediction(&angle, &flag1);
    myled1.write(0);
    printf("Close Mode 1 !\n\n");
}
//RPC Function
void gestureMode(Arguments *in, Reply *out) {
    printf("Enter Mode 1!\n\n");
    mode = 1;
    uLCD_thread.start(LCD);
    myled1.write(1);
    gesture_thread.start(gestureUI);
}

//Mode 2///////////////////////////

double calculateTheta(double a1, double a2, double a3, double b1, double b2, double b3) {
    double len_x = sqrt(a1 * a1 + a2 * a2 + a3 * a3);
    double len_y = sqrt(b1 * b1 + b2 * b2 + b3 * b3);
    int dot = a1 * b1 + a2 * b2 + a3 * b3;
    double cosin = dot / (len_x * len_y);
    double theta = acos(cosin) * 180.0 / PI;
    return theta;
}

//Detect Function
void detecting(int* threshold) {
    printf("\nPlace your mbed on table...\n");
    printf("Wait until the led3 light...\n");
    ThisThread::sleep_for(5s);
    BSP_ACCELERO_AccGetXYZ(DataXYZ);
    double b1 = DataXYZ[0];
    double b2 = DataXYZ[1];
    double b3 = DataXYZ[2];
    myled3.write(1);
    printf("\nStart to detect the tilt angle!\n");
    printf("The threshold angle: %d\n", *threshold);
    int num = 0;
    ThisThread::sleep_for(2s);
    while(flag2 < 10) {
        BSP_ACCELERO_AccGetXYZ(DataXYZ);
        theta = calculateTheta(DataXYZ[0], DataXYZ[1], DataXYZ[2], b1, b2, b3);
        printf("\nNum: %d\nNow tilt angle: %lf\n\n", num++, theta);
        if(theta >= (*threshold)) {
            flag2++;
            publish_message(&mode);
        }
        ThisThread::sleep_for(2s);
    }
}
//Thread Function
void detectAngle(int* threshold) {
    detecting(threshold);
    myled2.write(0);
    myled3.write(0);
    printf("\nClose Mode 2 !\n");    
}
//RPC Function
void detectionMode(Arguments *in, Reply *out) {
    printf("Enter Mode 2!\n");
    flag2 = 0;
    myled2.write(1);
    threshold = in->getArg<double>();
    mode = 2;
    detect_thread.start(callback(&detectAngle, &threshold));
    while(1) {
        if(flag2 == 10) {
            break;
        } 
        ThisThread::sleep_for(500ms);
    }
    ThisThread::sleep_for(2s);
    detect_thread.join();
    printf("\nUse \" /closeAll/run \" to close the program !\n");
}

//Closed/////////////////////////////
void closeAll(Arguments *in, Reply *out) {
    closed = true;
}


    