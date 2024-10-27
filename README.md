# Vehicle Detection, Tracking, and Future Predictions: The Bengaluru Mobility Challenge

This project is focused on addressing the traffic management problem in Bengaluru through vehicle detection, tracking, and prediction methodologies. It was developed for a hackathon co-organized by the Bengaluru Traffic Police, the Centre for Data for Public Good, and the Indian Institute of Science (IISc).

## Overview

The project aims to:

- Detect and classify vehicles moving in multiple directions on different road segments.
- Track vehicle movements between designated points on the road.
- Count vehicles and predict traffic flow using advanced tracking and forecasting techniques.

This repository includes the code and dataset preprocessing steps used to build and evaluate the detection and prediction models, along with analysis for optimizing traffic flow and congestion management in Bengaluru.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Vehicle Detection and Classification**: Utilizes object detection models to classify vehicles (e.g., cars, bikes, buses) into types.
- **Tracking and Direction Analysis**: Tracks vehicle movements across six directions, classifying direction and counting vehicle types per segment.
- **Forecasting Traffic Flow**: Employs the ARIMA model to predict vehicle counts at 1.5-minute intervals, aiding in traffic control and congestion prediction.
- **Enhanced Tracking Accuracy**: Leverages Kalman filters for improved vehicle tracking precision across frames.

## Installation

1. **Clone the repository:**

    \`\`\`bash
    git clone https://github.com/nabayansaha/Vehicle-Detection-Tracking-and-Future-Predictions-The-Bengaluru-Mobility-Challenge.git
    \`\`\`

2. **Navigate to the project directory:**

    \`\`\`bash
    cd Vehicle-Detection-Tracking-and-Future-Predictions-The-Bengaluru-Mobility-Challenge
    \`\`\`

3. **Set up the environment:**

    \`\`\`bash
    conda activate test_env_gpu
    \`\`\`

## Usage

1. Prepare your dataset by organizing vehicle images according to the specified directory structure.
2. Run the detection and tracking models to classify vehicles and analyze their directions.
3. Output the vehicle counts and directions to a CSV file.
4. Use the ARIMA model to predict vehicle flow based on tracked data.

For further details, refer to the scripts within each directory.

## Project Structure

\`\`\`plaintext
Vehicle-Detection-Tracking-and-Future-Predictions-The-Bengaluru-Mobility-Challenge
├── data/                 # Dataset directory
├── models/               # Pre-trained and custom models
├── notebooks/            # Jupyter notebooks for experimentation
├── src/                  # Core code for detection, tracking, and prediction
└── README.md             # Project overview and instructions
\`\`\`

## Methodology

1. **Vehicle Classification**: Detect and classify vehicles using object detection models.
2. **Tracking and Counting**: Track vehicle movement and count each type of vehicle for designated road segments.
3. **Future Prediction**: Apply ARIMA to forecast traffic volumes at intervals to support traffic management.

## Future Work

- Expand the vehicle classification to identify more vehicle types.
- Improve prediction accuracy by experimenting with other time-series models.
- Extend the project to provide real-time traffic analytics.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request.

## License

This project is licensed under the 
