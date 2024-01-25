#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import time
from datetime import datetime

import cv2
import numpy as np
import schedule
import tinytuya
from dotenv import load_dotenv
from influxdb import InfluxDBClient

# config logging
logging.basicConfig(filename='science_fair.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

influxdb_ip = os.getenv('influxdb_ip')
influxdb_port = os.getenv('influxdb_port')
influxdb_username = os.getenv('influxdb_username')
influxdb_password = os.getenv('influxdb_password')
influxdb_database_name = os.getenv('influxdb_database_name')

# tuya api
local_device_id = os.getenv('local_device_id')
local_device_ip = os.getenv('local_device_ip')
local_device_key = os.getenv('local_device_key')
local_device_version = os.getenv('local_device_version')

work_directory = os.getenv('work_directory')
rtsp_url = os.getenv('rtsp_url')

# duration interval in minutes
duration_picture = 10
# duration interval in seconds
duration_sensor = 30

json_data = {"measurement": "sensor_data", "tags": {}, "time": time.strftime('%Y-%m-%dT%H:%M:%S'), "fields": {}}

# use this monitor https://geekness.eu/python-tinytuya-temperature-monitor
# see lookup table inside for particular metrics
temperature = '1'
humidity = '2'
light_intensity = '16'

# Kernal to be used for strong laplace filtering
kernal_stkernal_strongrong = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]],
    dtype=np.float32)

# Kernal to be used for weak laplace filtering
kernal_weak = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]],
    dtype=np.float32)

output_image = (lambda n, v: cv2.imwrite(work_directory + n + '.png', v))


def get_temperature_sensor_data() -> dict:
    c = tinytuya.OutletDevice(dev_id=local_device_id, address=local_device_ip, version=local_device_version,
                              local_key=local_device_key)
    status = c.status()
    dps = status['dps']

    return {
        "temperature": dps[temperature],
        "humidity": dps[humidity],
        "light_intensity": dps[light_intensity]
    }


def count_bacteria_clones(image_path) -> int:
    name, ext = os.path.splitext(image_path)
    file_name = os.path.basename(name)
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Perform laplace filtering
    img_lap = cv2.filter2D(image, cv2.CV_32F, kernal_weak)
    img_sharp = np.float32(image) - img_lap

    # Save the sharpened image
    output_image(file_name + "-sharpened", img_sharp)

    # Convert to 8bits gray scale
    img_sharp = np.clip(image, 0, 255).astype('uint8')
    gray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)

    output_image(file_name + "-gray", gray)

    _, thresholded = cv2.threshold(gray, 5, 50, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = gray.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(name + "-" + str(len(contours)) + ".png", contour_image)

    #        cv2.imshow(f'Contours - Method {method}', contour_image)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of bacteria clones
    num_bacteria_clones = len(contours)

    return num_bacteria_clones


def save_image(timestamp):
    logging.info(f"Saving image for {timestamp}..")
    cap = cv2.VideoCapture(rtsp_url)
    ret, frame = cap.read()
    cv2.imshow('Capturing', frame)

    formatted_date = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d_%H-%M-%S')

    filename = work_directory + formatted_date + '.png'
    cv2.imwrite(filename, frame)
    logging.info(f"{filename} image saved!")
    cap.release()

    return filename


def take_picture():
    current_time = time.strftime('%Y-%m-%dT%H:%M:%S')
    # save file
    filename = save_image(current_time)
    json_data["tags"]["file_name"] = filename
    logging.info(f"saved image: {filename}")


def routine():
    # get the current timestamp
    current_time = time.strftime('%Y-%m-%dT%H:%M:%S')
    json_data['time'] = current_time


    # get temp/humid sensor
    json_data["fields"] = get_temperature_sensor_data()
    #    clones = count_bacteria_clones(filename)
    #    logging.info(f"Number of clones: {clones}")

    #    json_data["fields"]["counter"] = clones

    logging.info(f"json data: {json_data}")

    json = [json_data]
    client = InfluxDBClient(host=influxdb_ip, port=influxdb_port, username=influxdb_username,
                            password=influxdb_password, database=influxdb_database_name, gzip=True)
    client.write_points(json)
    client.close()


# Schedule your function to run every 10 minutes
schedule.every(duration_sensor).seconds.do(routine)
schedule.every(duration_picture).minutes.do(take_picture)

if __name__ == '__main__':
    logging.info("app starts!")
    routine()
    take_picture()
    logging.info("first observation completes.  now let's keep observing.")

    while True:
        schedule.run_pending()
        time.sleep(1)
