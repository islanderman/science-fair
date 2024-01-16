# Remi's Science Fair - Fix Second Rule Observation

## Overview
This is to help Remi's science project - Fix Second Rule for better data collection.

### Requirement
1. one machine to run this script continuously;
2. [temperature monitor](https://geekness.eu/python-tinytuya-temperature-monitor)
3. a camera that works with [open-cv](https://docs.opencv.org/)

## Details
The goal is to collect all related data in [influxdb](https://www.influxdata.com/) and observe in [grafana](https://grafana.com/docs/grafana/latest/getting-started/get-started-grafana-influxdb/) when everything is completed.  Ideally we should be able to observe the number of clones increase with timestamp, temperature, moisture level and so on.

This process is executed with schedule.

### Temperature and Moisture Gauging (Tuya Integration)
We purchased a [temperature monitor](https://geekness.eu/python-tinytuya-temperature-monitor) that we were able to retrieve the following data.

### Image
In this version we use a very simple camera with rtsp protocol, so that we are able to save images locally.  With the image, we process through opencv to get the number of contours, which is the experiment result.

All images are saved locally on the machine.

## Further Enhancement
Currently number of contours needs additional tweaking.  This does not work properly till we have accurate sample pictures for development purposes.