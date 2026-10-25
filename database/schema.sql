-- Automobile Efficiency Regressor & Vehicle Class Profiler
-- Database Schema for MySQL / MariaDB

CREATE DATABASE IF NOT EXISTS automobile_ml;
USE automobile_ml;

-- Table: predictions
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    cylinders INT NOT NULL,
    displacement FLOAT NOT NULL,
    horsepower FLOAT NOT NULL,
    weight FLOAT NOT NULL,
    acceleration FLOAT NOT NULL,
    model_year INT NOT NULL,
    origin VARCHAR(32) NOT NULL,
    predicted_mpg FLOAT NOT NULL,
    efficiency_category VARCHAR(64) NOT NULL,
    cluster_id INT NOT NULL,
    profile_name VARCHAR(128) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: model_metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(128) NOT NULL,
    mae FLOAT NOT NULL,
    mse FLOAT NOT NULL,
    rmse FLOAT NOT NULL,
    r2 FLOAT NOT NULL,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: cluster_profiles
CREATE TABLE IF NOT EXISTS cluster_profiles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    cluster_id INT NOT NULL UNIQUE,
    profile_name VARCHAR(128) NOT NULL,
    average_mpg FLOAT NOT NULL,
    average_weight FLOAT NOT NULL,
    average_horsepower FLOAT NOT NULL,
    average_displacement FLOAT NOT NULL,
    average_cylinders FLOAT NOT NULL,
    sample_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
