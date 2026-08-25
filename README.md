# Vision Traffic Analytics

Vision Traffic Analytics is a computer vision system designed to analyze video footage and detect the movement of people and vehicles across a defined counting line.

The system processes video input, detects relevant objects, tracks their movement, and determines whether they are entering or exiting an area. The resulting counts can be used to determine the number of people and vehicles currently present.

## Features

- Detect people and vehicles in video footage
- Track detected objects across video frames
- Count objects entering and exiting an area
- Define a custom virtual counting line
- Calculate the current number of objects present
- Process pre-recorded video footage
- Designed as a modular computer vision service

## Project Structure

```text
vision-traffic-analytics/
│
├── data/
│   ├── ground_truth/
│   └── videos/
│       ├── cars/
│       └── people/
│
├── src/
│   └── vision_traffic_analytics/
│
├── tests/
│
├── tools/
│   └── line_selector.py
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── uv.lock