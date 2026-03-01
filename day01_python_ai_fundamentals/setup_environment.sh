#!/bin/bash

# Day 01 – Environment Setup Script
# AlgoProfessor AI Internship

echo "Starting environment setup..."

# Upgrade pip
python -m pip install --upgrade pip

# Install required libraries
pip install -r requirements.txt

# Create outputs directory if not exists
mkdir -p outputs

echo "Environment setup completed successfully."
