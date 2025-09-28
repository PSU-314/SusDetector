# Face Recognition App

A Flask-based web application for face recognition in videos and photos. It detects the first appearance of known persons in a video using binary search and identifies photos containing known faces.

## Features
- **Video Processing**: Upload reference images and a video to detect the first frame where each person appears, with timestamps and thumbnails.
- **Photo Matching**: Upload reference images and a folder of photos to find photos containing known persons.
- **Clear Data**: Reset all uploaded and generated files.

## Prerequisites
- Python 3.8–3.11
- Git
- A video file (e.g., MP4) for default video processing (not included in repo due to size).

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/PSU-314/face-recognition-app.git
   cd face-recognition-app