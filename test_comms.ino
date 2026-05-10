#include <Encoder.h>

float fader1 = 0;
float resultFader1 = 0;
float fader2 = 0;
float resultFader2 = 0;
float fader3 = 0;
float resultFader3 = 0;
float fader4 = 0;
float resultFader4 = 0;

const int button1Pin = 2;
const int button2Pin = 3;

const byte rxPin = 4;
const byte txPin = 5;

// Encoder pins
const int enc1PinA = 8;
const int enc1PinB = 9;
const int enc2PinA = 6;
const int enc2PinB = 7;

// Create encoder objects
Encoder dial1(enc1PinA, enc1PinB);
Encoder dial2(enc2PinA, enc2PinB);

// Track last states
int lastButton1State = HIGH;
int lastButton2State = HIGH;

long lastEnc1Pos = 50;
long lastEnc2Pos = 50;

float mapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void setup() {
  Serial.begin(115200);

  pinMode(rxPin, INPUT);
  pinMode(txPin, OUTPUT);

  pinMode(button1Pin, INPUT_PULLUP);
  pinMode(button2Pin, INPUT_PULLUP);
}

void loop() {
  // Read faders once
  int raw1 = analogRead(A1);
  int raw2 = analogRead(A2);
  int raw3 = analogRead(A3);
  int raw4 = analogRead(A4);

  // Read buttons
  int button1State = digitalRead(button1Pin);
  int button2State = digitalRead(button2Pin);

  // Read encoders
  long newEnc1Pos = dial1.read();
  long newEnc2Pos = dial2.read();

  // -------- FADER 1 --------
  if (abs(raw1 - fader1) > 2) {
    fader1 = raw1;

    if (fader1 > 625) {
      resultFader1 = mapFloat(fader1, 625, 1023, 1, 10);
    } else {
      resultFader1 = mapFloat(fader1, 535, 615, 0, 1);
    }

    Serial.print("fader1:");
    Serial.println(resultFader1, 2);
  }

  // -------- FADER 2 --------
  if (abs(raw2 - fader2) > 2) {
    fader2 = raw2;

    if (fader2 > 629) {
      resultFader2 = mapFloat(fader2, 629, 1023, 1, 10);
    } else {
      resultFader2 = mapFloat(fader2, 540, 615, 0, 1);
    }

    Serial.print("fader2:");
    Serial.println(resultFader2, 2);
  }

  // -------- FADER 3 --------
  if (abs(raw3 - fader3) > 2) {
    fader3 = raw3;

    if (fader3 > 601) {
      resultFader3 = mapFloat(fader3, 601, 1023, 1, 10);
    } else {
      resultFader3 = mapFloat(fader3, 519, 600, 0, 1);
    }

    Serial.print("fader3:");
    Serial.println(resultFader3, 2);
  }

  // -------- FADER 4 --------
  if (abs(raw4 - fader4) > 2) {
    fader4 = raw4;

    if (fader4 > 641) {
      resultFader4 = mapFloat(fader4, 641, 1023, 1, 10);
    } else {
      resultFader4 = mapFloat(fader4, 547, 635, 0, 1);
    }

    Serial.print("fader4:");
    Serial.println(resultFader4, 2);
  }

// -------- BUTTON 1 --------
if (button1State != lastButton1State) {
  lastButton1State = button1State;

  if (button1State == LOW) {
    dial1.write(0);
    lastEnc1Pos = 0;

    // Serial.println("button1:pressed");
    Serial.println("encoder1:0.00");
  } else {
    // Serial.println("button1:released");
  }
}

// -------- BUTTON 2 --------
if (button2State != lastButton2State) {
  lastButton2State = button2State;

  if (button2State == LOW) {
    dial2.write(0);
    lastEnc2Pos = 0;

    // Serial.println("button2:pressed");
    Serial.println("encoder2:0.00");
  } else {
    // Serial.println("button2:released");
  }
}

// -------- ENCODER 1 --------
if (newEnc1Pos != lastEnc1Pos) {

  // Clamp encoder position to -50 to 50
  if (newEnc1Pos > 25) {
    newEnc1Pos = 25;
    dial1.write(25);
  } 
  else if (newEnc1Pos < -25) {
    newEnc1Pos = -25;
    dial1.write(-25);
  }

  float resultEnc1 = mapFloat(newEnc1Pos, -25, 25, -1, 1);

  Serial.print("encoder1:");
  Serial.println(resultEnc1, 2);

  lastEnc1Pos = newEnc1Pos;
}
// -------- ENCODER 2 --------
if (newEnc2Pos != lastEnc2Pos) {

  if (newEnc2Pos > 25) {
    newEnc2Pos = 25;
    dial2.write(25);
  } 
  else if (newEnc2Pos < -25) {
    newEnc2Pos = -25;
    dial2.write(-25);
  }

  float resultEnc2 = mapFloat(newEnc2Pos, -25, 25, -1, 1);

  Serial.print("encoder2:");
  Serial.println(resultEnc2, 2);

  lastEnc2Pos = newEnc2Pos;
}
  delay(5);
}