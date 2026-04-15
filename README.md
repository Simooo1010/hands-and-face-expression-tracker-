# Real-Time Hand Tracking Prototype

A lightweight, purely client-side web application for testing real-time hand and finger landmark detection using a computer webcam.

## Features
- Minimalist UI focused entirely on drawing the hand skeleton, face mesh, and analyzing facial expressions.
- Uses **MediaPipe Tasks Vision** (via CDN - no local Python dependencies required).
- Tracks up to 2 hands and 1 face in real-time.
- Displays standard metrics: FPS, number of hands/faces detected, and Handedness (Left/Right) with confidence scores.
- Parses facial blendshapes to recognize expressions like Smiling, Blinking, Brows Raised, etc.
- Dynamic visual toggling of points (landmarks) and lines (connections) for clear testing.
- Camera feed automatically mirrors horizontally for natural interaction.

## Prerequisites
- A modern web browser (Chrome, Firefox, Edge, Safari).
- An active webcam.
- A local HTTP server is required because browsers block webcam access for local `file:///` URLs due to security constraints.

## Setup & Running Instructions

1. **Navigate to the Project Folder** (where this file is located).

2. **Start a Local Development Server**:
   You can use Node.js or Python to serve the directory quickly.

   *If you have Python installed:*
   ```bash
   python -m http.server 8000
   ```

   *If you have Node.js / NPM installed:*
   ```bash
   npx http-server
   ```

3. **Open the App in your Browser**:
   Navigate to the local address provided by the server, typically `http://localhost:8000` or `http://127.0.0.1:8080`.

4. **Grant Webcam Permissions**:
   The browser will prompt you allowing access to your camera. The app will only start the model and rendering once permission is granted.

## Troubleshooting
- **No Video / Blank Screen**: Ensure no other application (like Zoom or Teams) is using the webcam. Check the browser permissions (click the lock icon on the URL bar) to ensure camera access is allowed on localhost.
- **Model Loading Slow**: It might take a few seconds initially to download the MediaPipe AI model via the CDN. Ensure you have an active internet connection.
