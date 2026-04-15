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
        document.getElementById("start-btn").disabled = false;
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
        const t = translations[window.currentLang || 'en'];
        loadingMsg.innerText = t.errCamera;
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

const translations = {
    en: {
        title: "Hand Tracking Tool",
        description: "Explore real-time hand and face tracking using advanced AI. See landmarks, gestures, and expressions in high performance.",
        infoCamera: "Camera access is required for real-time tracking. No data is sent to any server; all processing happens locally on your device.",
        startBtn: "Enable Camera & Start",
        labelHands: "Hands Detected:",
        labelHandedness: "Handedness:",
        labelFaces: "Faces Detected:",
        labelExpressions: "Expressions:",
        labelHandLandmarks: "Show Hand Landmarks",
        labelHandConnections: "Show Hand Connections",
        labelFaceLandmarks: "Show Face Landmarks",
        loading: "Initializing AI Models...",
        errCamera: "Camera access denied or unavailable. Please grant permissions."
    },
    it: {
        title: "Strumento di Tracciamento Mani",
        description: "Esplora il tracciamento in tempo reale di mani e viso con intelligenza artificiale avanzata. Visualizza punti di riferimento, gesti ed espressioni.",
        infoCamera: "L'accesso alla telecamera è necessario per il tracciamento in tempo reale. Nessun dato viene inviato a server; l'elaborazione avviene localmente.",
        startBtn: "Abilita Fotocamera e Inizia",
        labelHands: "Mani Rilevate:",
        labelHandedness: "Lateralità:",
        labelFaces: "Visi Rilevati:",
        labelExpressions: "Espressioni:",
        labelHandLandmarks: "Mostra Punti Reperere Mani",
        labelHandConnections: "Mostra Connessioni Mani",
        labelFaceLandmarks: "Mostra Punti Reperere Viso",
        loading: "Inizializzazione Modelli IA...",
        errCamera: "Accesso alla telecamera negato o non disponibile."
    },
    es: {
        title: "Herramienta de Seguimiento de Manos",
        description: "Explora el seguimiento en tiempo real de manos y cara usando inteligencia artificial avanzada.",
        infoCamera: "El acceso a la cámara es necesario para el seguimiento. Ningún dato se envía a un servidor; todo procesado localmente.",
        startBtn: "Habilitar Cámara y Comenzar",
        labelHands: "Manos Detectadas:",
        labelHandedness: "Lateralidad:",
        labelFaces: "Rostros Detectados:",
        labelExpressions: "Expresiones:",
        labelHandLandmarks: "Mostrar Puntos Manos",
        labelHandConnections: "Mostrar Conexiones Manos",
        labelFaceLandmarks: "Mostrar Puntos Rostro",
        loading: "Inicializando Modelos de IA...",
        errCamera: "Acceso a la cámara denegado o no disponible."
    },
    fr: {
        title: "Outil de Suivi des Mains",
        description: "Explorez le suivi en temps réel des mains et du visage à l'aide d'une IA avancée.",
        infoCamera: "L'accès à la caméra est requis pour le suivi en temps réel. Aucune donnée n'est envoyée à un serveur.",
        startBtn: "Activer la Caméra et Démarrer",
        labelHands: "Mains Détectées:",
        labelHandedness: "Latéralité:",
        labelFaces: "Visages Détectés:",
        labelExpressions: "Expressions:",
        labelHandLandmarks: "Afficher Points Repères Mains",
        labelHandConnections: "Afficher Connexions Mains",
        labelFaceLandmarks: "Afficher Points Repères Visage",
        loading: "Initialisation des Modèles IA...",
        errCamera: "Accès à la caméra refusé ou indisponible."
    }
};

window.setLanguage = function (lang) {
    document.querySelectorAll('.lang-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.lang-btn[data-lang="${lang}"]`).classList.add('active');

    const t = translations[lang];
    document.getElementById('title').innerText = t.title;
    document.getElementById('description').innerText = t.description;
    document.getElementById('info-camera').innerText = t.infoCamera;
    document.getElementById('start-btn').innerText = t.startBtn;
    document.getElementById('label-hands').innerText = t.labelHands;
    document.getElementById('label-handedness').innerText = t.labelHandedness;
    document.getElementById('label-faces').innerText = t.labelFaces;
    document.getElementById('label-expressions').innerText = t.labelExpressions;
    document.getElementById('label-hand-landmarks').innerText = t.labelHandLandmarks;
    document.getElementById('label-hand-connections').innerText = t.labelHandConnections;
    document.getElementById('label-face-landmarks').innerText = t.labelFaceLandmarks;

    if (loadingMsg.innerText.includes("Initializing") || loadingMsg.innerText.includes("Inizializzazione") || loadingMsg.innerText.includes("Inicializando") || loadingMsg.innerText.includes("Initialisation")) {
        loadingMsg.innerText = t.loading;
    }

    // Store current lang for errors
    window.currentLang = lang;
};

document.getElementById('start-btn').addEventListener('click', () => {
    document.getElementById('splash-screen').style.display = 'none';
    document.getElementById('app-container').style.display = 'flex';
    startCamera();
});

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
    if (handResults && handResults.landmarks && handResults.landmarks.length > 0) {
        handCountElement.innerText = handResults.landmarks.length;
        let handednessArr = [];

        for (let i = 0; i < handResults.landmarks.length; i++) {
            const landmarks = handResults.landmarks[i];

            if (handResults.handednesses && handResults.handednesses[i]) {
                const category = handResults.handednesses[i][0];
                const type = category.categoryName;
                let confidence = Math.round(category.score * 100);
                handednessArr.push(`Hand ${i + 1}: ${type} (${confidence}%)`);
            }

            if (toggleConnections.checked) {
                drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
                    color: "rgba(0, 255, 0, 0.7)",
                    lineWidth: 4
                });
            }
            if (toggleLandmarks.checked) {
                drawingUtils.drawLandmarks(landmarks, {
                    color: "rgba(255, 0, 0, 0.9)",
                    lineWidth: 2,
                    radius: 4,
                    fillColor: "rgba(255, 165, 0, 0.8)"
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
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030", lineWidth: 2 });
                // Right Eyebrow
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030", lineWidth: 2 });
                // Left Eye
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30", lineWidth: 2 });
                // Left Eyebrow
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30", lineWidth: 2 });
                // Face Oval
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0", lineWidth: 2 });
                // Lips
                drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#FF3030", lineWidth: 2 });
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
// Start application by disabling button until models are ready
document.getElementById("start-btn").disabled = true;
window.currentLang = 'en';
initializeModels();
