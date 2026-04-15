import {
    HandLandmarker,
    FaceLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const loadingMsg = document.getElementById("loading");
const infoDiv = document.getElementById("info");
const fpsElement = document.getElementById("fps");
const handCountElement = document.getElementById("hand-count");
const handednessElement = document.getElementById("handedness");
const faceCountElement = document.getElementById("face-count");
const expressionsElement = document.getElementById("expressions");

const toggleLandmarks = document.getElementById("toggle-landmarks");
const toggleConnections = document.getElementById("toggle-connections");
const toggleFaceLandmarks = document.getElementById("toggle-face-landmarks");

let handLandmarker = undefined;
let faceLandmarker = undefined;
let runningMode = "VIDEO";
let lastVideoTime = -1;
let handResults = undefined;
let faceResults = undefined;

let frameCount = 0;
let lastFpsTime = performance.now();
let currentFps = 0;

// 1. Initialize the MediaPipe AI Models
async function initializeModels() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        
        // Load Hand Landmarker
        const handPromise = HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                delegate: "GPU"
            },
            runningMode: runningMode,
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        // Load Face Landmarker
        const facePromise = FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU"
            },
            outputFaceBlendshapes: true, // Required for evaluating expressions
            runningMode: runningMode,
            numFaces: 1,
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        [handLandmarker, faceLandmarker] = await Promise.all([handPromise, facePromise]);
        
        loadingMsg.style.display = "none";
        infoDiv.style.display = "block";
        
        // Start the camera after models are loaded
        startCamera();
    } catch (error) {
        console.error("Error loading MediaPipe models:", error);
        loadingMsg.innerText = "Error loading models. Check console.";
        loadingMsg.style.color = "#ff4d4d";
    }
}

// 2. Start Webcam Feed
async function startCamera() {
    const constraints = {
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user"
        }
    };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    } catch (err) {
        console.error("Error accessing webcam:", err);
        loadingMsg.style.display = "block";
        loadingMsg.innerText = "Camera access denied or unavailable. Please grant permissions.";
        loadingMsg.style.color = "#ff4d4d";
        infoDiv.style.display = "none";
    }
}

// Helper: Evaluate facial expressions based on blendshapes
function analyzeExpressions(blendshapes) {
    if (!blendshapes || blendshapes.length === 0) return "Neutral";
    
    // We only have 1 face configured
    const categories = blendshapes[0].categories;
    const expressions = [];
    
    // Create a dictionary for logic comparisons
    const scores = {};
    for (const shape of categories) {
        scores[shape.categoryName] = shape.score;
    }

    const threshold = 0.5;

    // Smile logic
    if (scores["mouthSmileLeft"] > threshold || scores["mouthSmileRight"] > threshold) {
        expressions.push("😊 Smiling");
    }
    
    // Mouth open logic
    if (scores["jawOpen"] > 0.4) {
        expressions.push("😮 Mouth Open");
    }
    
    // Blinking
    if (scores["eyeBlinkLeft"] > threshold && scores["eyeBlinkRight"] > threshold) {
         expressions.push("😑 Blinking (Both Eyes)");
    } else if (scores["eyeBlinkLeft"] > threshold) {
         expressions.push("😉 Wink (Left Eye)"); // Model tracks left/right from user perspective
    } else if (scores["eyeBlinkRight"] > threshold) {
         expressions.push("😉 Wink (Right Eye)");
    }

    // Eyebrows
    if (scores["browInnerUp"] > 0.6) {
        expressions.push("😲 Brows Raised");
    } else if (scores["browDownLeft"] > threshold || scores["browDownRight"] > threshold) {
        expressions.push("😠 Brows Furrowed");
    }

    if (expressions.length === 0) {
        return "Neutral";
    }

    return expressions.join("<br>");
}

const MAX_PREDICT_FRAMES = 60; // Predict for ~2 seconds if occluded (assuming ~30 fps)
let trackedHands = []; 
let handIdCounter = 0;

function calculateCentroid(landmarks) {
    let x = 0, y = 0, z = 0;
    for (let lm of landmarks) {
        x += lm.x; y += lm.y; z += lm.z;
    }
    const len = landmarks.length;
    return { x: x/len, y: y/len, z: z/len };
}

function updateTrackedHands(detectedLandmarks, detectedHandednesses) {
    const currentDetected = [];
    const numHands = detectedLandmarks ? detectedLandmarks.length : 0;
    
    for (let i = 0; i < numHands; i++) {
        // Deep copy landmarks so we can modify them during prediction
        const copiedLandmarks = detectedLandmarks[i].map(lm => ({x: lm.x, y: lm.y, z: lm.z}));
        currentDetected.push({
            landmarks: copiedLandmarks,
            centroid: calculateCentroid(copiedLandmarks),
            handedness: detectedHandednesses ? detectedHandednesses[i] : null,
            matched: false
        });
    }

    // Match existing tracked hands to detected hands
    for (let th of trackedHands) {
        let bestMatch = null;
        let bestDist = Infinity;
        
        for (let cd of currentDetected) {
            if (cd.matched) continue;
            const dx = th.centroid.x - cd.centroid.x;
            const dy = th.centroid.y - cd.centroid.y;
            const dist = dx*dx + dy*dy;
            if (dist < 0.1 && dist < bestDist) { // Allow matching within a reasonable distance
                bestDist = dist;
                bestMatch = cd;
            }
        }
        
        if (bestMatch) {
            bestMatch.matched = true;
            // Smooth velocity calculation to prevent jittering during prediction
            const newVx = bestMatch.centroid.x - th.centroid.x;
            const newVy = bestMatch.centroid.y - th.centroid.y;
            const newVz = bestMatch.centroid.z - th.centroid.z;
            th.velocity = {
                x: th.velocity.x * 0.5 + newVx * 0.5,
                y: th.velocity.y * 0.5 + newVy * 0.5,
                z: th.velocity.z * 0.5 + newVz * 0.5
            };
            th.centroid = bestMatch.centroid;
            th.landmarks = bestMatch.landmarks;
            th.handedness = bestMatch.handedness || th.handedness;
            th.lostFrames = 0;
            th.isPredicted = false;
        } else {
            // Hand lost, predict next position using velocity
            th.lostFrames++;
            if (th.lostFrames < MAX_PREDICT_FRAMES) {
                // Apply friction to velocity so it doesn't drift endlessly
                th.velocity.x *= 0.95;
                th.velocity.y *= 0.95;
                th.velocity.z *= 0.95;

                for (let lm of th.landmarks) {
                    lm.x += th.velocity.x;
                    lm.y += th.velocity.y;
                    lm.z += th.velocity.z;
                }
                th.centroid.x += th.velocity.x;
                th.centroid.y += th.velocity.y;
                th.centroid.z += th.velocity.z;
                th.isPredicted = true;
            }
        }
    }

    trackedHands = trackedHands.filter(th => th.lostFrames < MAX_PREDICT_FRAMES);

    for (let cd of currentDetected) {
        if (!cd.matched) {
            trackedHands.push({
                id: handIdCounter++,
                landmarks: cd.landmarks,
                centroid: cd.centroid,
                velocity: {x: 0, y: 0, z: 0},
                lostFrames: 0,
                handedness: cd.handedness,
                isPredicted: false
            });
        }
    }
}

// 3. Process video feed and draw
async function predictWebcam() {
    if (canvasElement.width !== video.videoWidth) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    let startTimeMs = performance.now();
    
    // Run prediction if we have a new frame
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        if (handLandmarker) {
            handResults = handLandmarker.detectForVideo(video, startTimeMs);
            updateTrackedHands(handResults?.landmarks, handResults?.handednesses);
        }
        if (faceLandmarker) {
            faceResults = faceLandmarker.detectForVideo(video, startTimeMs);
        }
    }

    // Calculate FPS
    frameCount++;
    if (startTimeMs - lastFpsTime >= 1000) {
        currentFps = Math.round((frameCount * 1000) / (startTimeMs - lastFpsTime));
        fpsElement.innerText = currentFps;
        frameCount = 0;
        lastFpsTime = startTimeMs;
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    const drawingUtils = new DrawingUtils(canvasCtx);

    // --- Process & Draw Hands ---
    if (trackedHands.length > 0) {
        let predictedCount = trackedHands.filter(h => h.isPredicted).length;
        let actualCount = trackedHands.length - predictedCount;
        handCountElement.innerHTML = `${actualCount} <span style="font-size:0.8em; color:gray;">(${predictedCount} predicted)</span>`;
        let handednessArr = [];
        
        for (let i = 0; i < trackedHands.length; i++) {
            const th = trackedHands[i];
            const landmarks = th.landmarks;
            
            if (th.handedness) {
                const category = th.handedness[0];
                const type = category.categoryName; 
                let confidence = Math.round(category.score * 100);
                const statusStr = th.isPredicted ? " <span style='color:orange;'>(Predicted)</span>" : "";
                handednessArr.push(`Hand ${i+1}: ${type} (${confidence}%)${statusStr}`);
            }

            if (toggleConnections.checked) {
                drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
                    color: th.isPredicted ? "rgba(0, 255, 255, 0.5)" : "rgba(0, 255, 0, 0.7)", 
                    lineWidth: 4
                });
            }
            if (toggleLandmarks.checked) {
                drawingUtils.drawLandmarks(landmarks, {
                    color: th.isPredicted ? "rgba(100, 200, 255, 0.6)" : "rgba(255, 0, 0, 0.9)", 
                    lineWidth: 2,
                    radius: th.isPredicted ? 3 : 4,
                    fillColor: th.isPredicted ? "rgba(100, 150, 255, 0.5)" : "rgba(255, 165, 0, 0.8)"
                });
            }
        }
        handednessElement.innerHTML = handednessArr.join("<br>");
    } else {
        handCountElement.innerText = "0";
        handednessElement.innerText = "None";
    }

    // --- Process & Draw Faces ---
    if (faceResults && faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
        faceCountElement.innerText = faceResults.faceLandmarks.length;
        
        for (const landmarks of faceResults.faceLandmarks) {
            if (toggleFaceLandmarks.checked) {
                // Tesselation (mesh over face)
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
                    color: "rgba(192, 192, 192, 0.3)", // Light silver, semi-transparent
                    lineWidth: 1
                });
                // Right Eye
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {color: "#FF3030", lineWidth: 2});
                // Right Eyebrow
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, {color: "#FF3030", lineWidth: 2});
                // Left Eye
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {color: "#30FF30", lineWidth: 2});
                // Left Eyebrow
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, {color: "#30FF30", lineWidth: 2});
                // Face Oval
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {color: "#E0E0E0", lineWidth: 2});
                // Lips
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {color: "#FF3030", lineWidth: 2});
            }
        }
        
        // Output Facial Expressions from blendshapes
        if (faceResults.faceBlendshapes) {
            const detectedExpressions = analyzeExpressions(faceResults.faceBlendshapes);
            expressionsElement.innerHTML = detectedExpressions;
        }
        
    } else {
        faceCountElement.innerText = "0";
        expressionsElement.innerText = "None";
    }
    
    canvasCtx.restore();

    window.requestAnimationFrame(predictWebcam);
}

// Start application
initializeModels();
