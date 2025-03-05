#include "Wire.h"
#include "SparkFun_Displacement_Sensor_Arduino_Library.h" // Click here to get the library: http://librarymanager/All#SparkFun_Displacement_Sensor

///////////////////////
// NOTE: YOU NEED TO UPDATE THE FINGER MAPPINGS (A few lines down) AND SENSOR MAPPINGS (get_sensor_readings()) BEFORE RUNNING THIS CODE


ADS flexSensor1; // Create object of the ADS class


// Finger Mappings
int Finger1_i = 9;//inside
int Finger1_o = 4;//outside
int Finger1_cw = 7;//clockwise
int Finger1_cc = 6;//counter-clock


const int Active = 80; // PWM amount
const int Active_close = 40; // PWM amount when close to goal
int Active_cur_1_x = 0; // update this based on position error
int Active_cur_1_y = 0; // update this based on position error
const int close_bound = 3; // degrees to be considered close

float x_1_new;
float y_1_new;

float x_1_filt;

float x_1_filt_prev = 0;

float v_1 = 0;

float v_1_prev = 0;

float a_1 = 0;

bool contact_1 = false;

bool target_updated_1 = false;


float target_1_x = -30;
float target_1_y = 0;
int targets_reached = 0; // zero initialized

float x_1_err = 0;
float y_1_err = 0;

// Movement commands
byte set_1_i;
byte set_1_o;
byte set_1_cw;
byte set_1_cc;

// FOR Gaussian filter

const int numPoints = 15;       // Number of points for the Gaussian mean
float sigma = 3.0; 
float data_x1[numPoints] = {0};    // Array to store the last 15 data points
float gaussianWeights[numPoints];  // Array to store Gaussian weights
int currentIndex = 0;           // Index to keep track of the most recent data point



void setup() {

    Wire.begin();
    Serial.begin(2000000);
    // Serial.println("Starting");
    initialize_sensors();
    initialize_actuators();
    compute_gaussian_weights();

}

void loop() {

  int in = 30;
  int out = -25;
  int left = 20;
  int right = -20;

  // change the waypoints for different motions!
  int pos1[5] = {30, -30, 1000, 0, 1000}; // x1, y1,hold time (ms), check for contant [bool], wait time (ms)
  int pos2[5] = {30, 30, 1000, 0, 10000}; // x1, y1, hold time (ms), check for contant [bool], wait time (ms)
  int pos3[5] = {15, -30, 1000, 0, 1000}; // x1, y1,hold time (ms), check for contant [bool], wait time (ms)
  int pos4[5] = {15, 30, 1000, 0, 10000}; // x1, y1, hold time (ms), check for contant [bool], wait time (ms)
  int pos5[5] = {0, -30, 1000, 0, 1000}; // x1, y1,hold time (ms), check for contant [bool], wait time (ms)
  int pos6[5] = {0, 30, 1000, 0, 10000}; // x1, y1, hold time (ms), check for contant [bool], wait time (ms)
  int pos7[5] = {-15, -30, 1000, 0, 1000}; // x1, y1,hold time (ms), check for contant [bool], wait time (ms)
  int pos8[5] = {-15, 30, 1000, 0, 10000}; // x1, y1, hold time (ms), check for contant [bool], wait time (ms)
  int pos9[5] = {-30, -30, 1000, 0, 1000}; // x1, y1,hold time (ms), check for contant [bool], wait time (ms)
  int pos10[5] = {-30, 30, 1000, 0, 10000}; // x1, y1, hold time (ms), check for contant [bool], wait time (ms)
  int pos11[5] = {-30, -30, 1000, 0, 1000}; // x1, y1, hold time (ms), check for contant [bool], wait time (ms)
  int pos12[5] = {30, -30, 1000, 0, 1000}; // x1, y1,hold time (ms), check for contant [bool], wait time (ms)
  int pos13[5] = {30, 30, 1000, 0, 0}; // x1, y1, hold time (ms), check for contant [bool], wait time (ms)
  int pos14[5] = {-30, 30, 1000, 0, 1000000}; // x1, y1, hold time (ms), check for contant [bool], wait time (ms)



  int num_waypoints = 14;
  int waypoints[num_waypoints] = {pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, pos11, pos12, pos13, pos14};

  for (int i=0; i<num_waypoints; i++){
    Serial.print("Waypoint: ");
    Serial.println(String(i));
    targets_reached = 0;
    int* wp = waypoints[i];
    target_1_x = wp[0];
    target_1_y = wp[1];
    long hold_ms = wp[2];
    bool check_contact = wp[3];
    long wait_ms = wp[4];
    while(targets_reached < 2) {
      sense_and_move();
      calculate_filtered_vals();
      calculate_derivatives();

      // Printing
      String positions = String(x_1_new) + "," + String(y_1_new);
      String targets = String(target_1_x) + "," + String(target_1_y);
      String pwms = String(set_1_i) + "," + String(set_1_o) + "," + String(set_1_cc) + "," + String(set_1_cw);
      String output = String(micros()) + "," + positions + "," + targets + "," + pwms;
      Serial.println(output);


      if (check_contact) {
        check_for_contact();
        update_targets_for_contact();
      }
    }
    if (hold_ms > 0) { // code to hold position 
      long start = millis();
      while ((millis()-start) < hold_ms) {
        sense_and_move();
        calculate_filtered_vals();
        calculate_derivatives();
        // String output = String(micros()) + "," + String(x_2_new) + ", " + String(y_2_new) + ", " + String(target_2_x) + ", " + String(target_2_y); // + ", " + String(x_2_filt)+ ", " + String(v_2)+ ", " + String(a_2)+ ", " + String(contact_2);
        // Serial.print(output);
        
        // Printing
        String positions = String(x_1_new) + "," + String(y_1_new);
        String targets = String(target_1_x) + "," + String(target_1_y);
        String pwms = String(set_1_i) + "," + String(set_1_o) + "," + String(set_1_cc) + "," + String(set_1_cw);
        String output = String(micros()) + "," + positions + "," + targets + "," + pwms;
        Serial.println(output);

      }
  
    }
    // Cool down time
    long start = millis();
    while ((millis()-start) < wait_ms) {
      delay(20); // wait 20 ms
      // Printing
      String positions = String(x_1_new) + "," + String(y_1_new);
      String targets = String(target_1_x) + "," + String(target_1_y);
      String pwms = String(set_1_i) + "," + String(set_1_o) + "," + String(set_1_cc) + "," + String(set_1_cw);
      String output = String(micros()) + "," + positions + "," + targets + "," + pwms;
      Serial.println(output);
    }
  }


}



void initialize_sensors() {
    if (flexSensor1.begin() == false) {
      Serial.println(F("Sensor 1 not detected. Check wiring. Freezing..."));
      while (1);
    }
    return;
}


void initialize_actuators() {
    pinMode(Finger1_o, OUTPUT);
    pinMode(Finger1_i, OUTPUT);
    pinMode(Finger1_cc, OUTPUT);
    pinMode(Finger1_cw, OUTPUT);

    return;
}


void sense_and_move() {
    get_sensor_readings(); // ~3.5 ms

    calculate_errors(); // update position errors // ~ 0.06 ms


    check_target_reached();
    set_movement_speeds(); // ~ 0.06 ms
    
    actuate(); // ~0.33 ms in addition to actuation time
}

void get_sensor_readings() {

  if (flexSensor1.available())
  {
    x_1_new = flexSensor1.getY(); 
    y_1_new = flexSensor1.getX();
  }

  return;
}

void calculate_errors() {
  x_1_err = x_1_new - target_1_x;
  y_1_err = y_1_new - target_1_y;

  return;
}


void check_target_reached() {
  targets_reached = 0;
  if (abs(x_1_err) < close_bound) {targets_reached++;}
  if (abs(y_1_err) < close_bound) {targets_reached++;}
  return;

}

void set_movement_speeds() {
    Active_cur_1_x = (abs(x_1_err) < close_bound) ? Active_close : Active;
    Active_cur_1_y = (abs(y_1_err) < close_bound) ? Active_close : Active;
    return;
}

void actuate() {
    set_1_i = (x_1_new < target_1_x) ? Active_cur_1_x : 0;
    set_1_o = (x_1_new > target_1_x) ? Active_cur_1_x : 0;
    set_1_cc = (y_1_new > target_1_y) ? Active_cur_1_y : 0;
    set_1_cw = (y_1_new < target_1_y) ? Active_cur_1_y : 0;




    analogWrite(Finger1_i, set_1_i);
    analogWrite(Finger1_o, set_1_o);
    analogWrite(Finger1_cc, set_1_cc);
    analogWrite(Finger1_cw, set_1_cw);
    delay(50);
    analogWrite(Finger1_i, 0);
    analogWrite(Finger1_o, 0);
    analogWrite(Finger1_cc, 0);
    analogWrite(Finger1_cw, 0);
}


void compute_gaussian_weights() {
  float sum = 0.0;

  for (int i = 0; i < numPoints; i++) {
    float x = (float)(i - (numPoints - 1) / 2);
    gaussianWeights[i] = exp(-0.5 * sq(x / sigma));
    sum += gaussianWeights[i];
  }

  // Normalize the weights so that they sum to 1
  for (int i = 0; i < numPoints; i++) {
    gaussianWeights[i] /= sum;
  }
}

// Update the data buffer with new data point
void updateBuffers() {
  data_x1[currentIndex] = x_1_new;
  currentIndex = (currentIndex + 1) % numPoints;

}

void calculate_filtered_vals() {
  updateBuffers();
  x_1_filt = 0.0;

  for (int i = 0; i < numPoints; i++) {
    int index = (currentIndex + i) % numPoints;
    x_1_filt += data_x1[index] * gaussianWeights[i];
  }


}

void calculate_derivatives() {

  float d_time = 0.055; // approx time between readings in seconds

  v_1 = (x_1_filt - x_1_filt_prev) / d_time;

  a_1 = (v_1 - v_1_prev) / d_time;


  x_1_filt_prev = x_1_filt;
  
  v_1_prev = v_1;
}

void check_for_contact() {
  float accel_contact_thresh = -10;
  float pos_contact_thresh = 0;
  if (a_1 < accel_contact_thresh && x_1_filt > pos_contact_thresh) {
    contact_1 = true;
    // Serial.println("CONTACT 1");
  }
  else {contact_1 = false;}
}

void update_targets_for_contact() {
  if (contact_1 && !target_updated_1) {
    target_1_x = x_1_new+10;
    target_updated_1 = true;
    Serial.println("new target_1_x = " + String(target_1_x));
  }
}
