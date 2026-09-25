/*
 * sensor_logger.ino  -  data collection sketch (Arduino Uno)
 *
 * Reads the capacitive soil moisture sensor and the DHT11 and prints one
 * block per sample in the format parsed by src/data_collection.py:
 *
 *     Soil raw: 512
 *     Temperature: 29.4 C
 *     Humidity: 61.0 %
 *     ---
 *
 * The relay is held OFF the whole time: during data collection the team
 * waters by hand so that watering events are controlled and recorded.
 *
 * Library: "DHT sensor library" by Adafruit (+ "Adafruit Unified Sensor").
 *
 * IMPORTANT: SOIL_SAMPLES and the averaging below must match ml_controller.ino,
 * otherwise the controller sees differently-processed values than the model
 * was trained on.
 */

#include <DHT.h>

// ---- Pin configuration (confirm against your actual wiring) ----
#define DHT_PIN 4
#define DHT_TYPE DHT11
#define SOIL_PIN A0
#define RELAY_PIN 7
#define RELAY_ACTIVE_LOW true   // many 1-channel relay boards switch ON when IN is LOW

// ---- Sampling ----
#define SERIAL_BAUD 9600
#define SAMPLE_INTERVAL_MS 60000UL   // one reading per minute
#define SOIL_SAMPLES 10              // analogRead() averaged over this many reads
#define SOIL_SAMPLE_DELAY_MS 10

DHT dht(DHT_PIN, DHT_TYPE);

void relayOff() {
  digitalWrite(RELAY_PIN, RELAY_ACTIVE_LOW ? HIGH : LOW);
}

int readSoilRaw() {
  long sum = 0;
  for (int i = 0; i < SOIL_SAMPLES; i++) {
    sum += analogRead(SOIL_PIN);
    delay(SOIL_SAMPLE_DELAY_MS);
  }
  return (int)(sum / SOIL_SAMPLES);
}

void setup() {
  relayOff();                 // set the level before enabling the output
  pinMode(RELAY_PIN, OUTPUT);
  relayOff();

  Serial.begin(SERIAL_BAUD);
  dht.begin();
  delay(2000);                // DHT11 needs time after power-up
  Serial.println("# sensor_logger started");
}

void loop() {
  int soilRaw = readSoilRaw();
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();   // Celsius

  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("# DHT11 read failed");     // the Python logger skips this sample
  } else {
    Serial.print("Soil raw: ");
    Serial.println(soilRaw);
    Serial.print("Temperature: ");
    Serial.print(temperature, 1);
    Serial.println(" C");
    Serial.print("Humidity: ");
    Serial.print(humidity, 1);
    Serial.println(" %");
    Serial.println("---");
  }

  delay(SAMPLE_INTERVAL_MS);
}
