"use client";

import { useEffect, useRef, useState } from "react";
import type { FaceLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision";
import {
  createFaceMeshIndices,
  estimateLandmarkConfidence,
  measureFace,
  estimateHeadRotation,
  type FaceMeasurements,
} from "@/lib/face-geometry";
import Glasses3D, { type FacePose } from "@/app/components/Glasses3D";

const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm";
const DEFAULT_MODEL_URL = "/frames/starter-wayfarer-glasses.glb";
const FRAMES = [
  { name: "Forma Classic", style: "Polished black acetate · Optical", price: "$245", shape: "Square", color: "#211f1c", mark: "FC", image: "/frames/black-frame.png" },
  { name: "The Riviera", style: "Polished titanium · Gold", price: "$310", shape: "Round", color: "#c7a35b", mark: "RI" },
  { name: "The Form", style: "Italian acetate · Tortoise", price: "$280", shape: "Square", color: "#77553b", mark: "FO" },
  { name: "The Solstice", style: "Featherweight metal · Silver", price: "$265", shape: "Oval", color: "#b9bdba", mark: "SO" },
];

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const listenerRef = useRef<{ track: MediaStreamTrack; handler: () => void } | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const animationRef = useRef<number | null>(null);
  const operationRef = useRef(0);
  const startingRef = useRef(false);
  const smoothedRef = useRef<Float32Array | null>(null);
  const poseRef = useRef<FacePose>(null);
  const faceMeshIndicesRef = useRef<Uint16Array | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [starting, setStarting] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [landmarkCount, setLandmarkCount] = useState(0);
  const [status, setStatus] = useState("Ready when you are");
  const [mode, setMode] = useState<"GPU" | "CPU" | null>(null);
  const [measurements, setMeasurements] = useState<FaceMeasurements | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(0);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [showUploadControls, setShowUploadControls] = useState(false);
  const [scale, setScale] = useState(1);
  const [offsetY, setOffsetY] = useState(0);
  const [depthOffset, setDepthOffset] = useState(0);
  const [rotationOffset, setRotationOffset] = useState({ pitch: 0, yaw: 0, roll: 0 });
  const [fullScreen, setFullScreen] = useState(false);
  const [captured, setCaptured] = useState<string | null>(null);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL_URL);
  const [modelStatus, setModelStatus] = useState("Loading included 3D frame…");
  const fileRef = useRef<HTMLInputElement>(null);
  const modelFileRef = useRef<HTMLInputElement>(null);

  function releaseCameraResources() {
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    landmarkerRef.current?.close(); landmarkerRef.current = null;
    const listener = listenerRef.current;
    if (listener) { listener.track.removeEventListener("ended", listener.handler); listenerRef.current = null; }
    streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current = null;
    smoothedRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    const canvas = canvasRef.current;
    canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  }

  async function startCamera() {
    if (startingRef.current || cameraActive) return;
    startingRef.current = true; setStarting(true); setError(""); setStatus("Requesting camera access…");
    const operation = ++operationRef.current;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access needs a modern browser over HTTPS or localhost.");
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      if (operation !== operationRef.current) { stream.getTracks().forEach((track) => track.stop()); return; }
      streamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      if (!track) throw new Error("No camera video track was provided.");
      const handleEnded = () => {
        if (operation !== operationRef.current) return;
        operationRef.current++; startingRef.current = false; poseRef.current = null; faceMeshIndicesRef.current = null; releaseCameraResources();
        setStarting(false); setCameraActive(false); setFaceDetected(false); setLandmarkCount(0); setMeasurements(null); setMode(null);
        setError("The camera disconnected. Reconnect it to continue."); setStatus("Camera disconnected");
      };
      track.addEventListener("ended", handleEnded, { once: true }); listenerRef.current = { track, handler: handleEnded };
      const video = videoRef.current; if (!video) throw new Error("Camera preview is unavailable.");
      video.srcObject = stream; await video.play(); if (operation !== operationRef.current) return;
      setStatus("Loading face tracking…");
      const { FilesetResolver, FaceLandmarker } = await import("@mediapipe/tasks-vision");
      const vision = await FilesetResolver.forVisionTasks(WASM_URL); if (operation !== operationRef.current) return;
      faceMeshIndicesRef.current = createFaceMeshIndices(
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      );
      const options = { baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" as const }, runningMode: "VIDEO" as const, numFaces: 1, minFaceDetectionConfidence: .5, minFacePresenceConfidence: .5, minTrackingConfidence: .5, outputFacialTransformationMatrixes: true };
      let landmarker: FaceLandmarker; let inference: "GPU" | "CPU" = "GPU";
      try { landmarker = await FaceLandmarker.createFromOptions(vision, options); }
      catch (gpuError) {
        console.warn("GPU face tracking unavailable; retrying on CPU.", gpuError); inference = "CPU";
        landmarker = await FaceLandmarker.createFromOptions(vision, { ...options, baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" } });
      }
      if (operation !== operationRef.current) { landmarker.close(); return; }
      landmarkerRef.current = landmarker; setMode(inference); setCameraActive(true); setStatus("Find a comfortable light");
      const overlay = canvasRef.current; const context = overlay?.getContext("2d");
      if (!overlay || !context) throw new Error("Face tracking overlay could not be initialized.");
      let lastVideoTime = -1, previousFrameTime = 0, lastMetricsUpdate = 0, hadFace = false, lostAt: number | null = null;
      function drawHints(points: NormalizedLandmark[], width: number, height: number, alpha: number) {
        const values = smoothedRef.current; if (!values) return;
        context!.save(); context!.globalAlpha = alpha; context!.fillStyle = "#d4b378";
        for (const index of [33, 133, 362, 263, 1, 6, 127, 356]) {
          const pos = index * 3; if (pos >= values.length) continue;
          context!.beginPath(); context!.arc(values[pos] * width, values[pos + 1] * height, index === 1 || index === 6 ? 2.6 : 2, 0, Math.PI * 2); context!.fill();
        }
        context!.restore(); void points;
      }
      function detect(frameTime: number) {
        if (operation !== operationRef.current) return;
        try {
          const current = videoRef.current, detector = landmarkerRef.current;
          if (current && detector && current.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && current.videoWidth > 0 && current.currentTime !== lastVideoTime) {
            lastVideoTime = current.currentTime; const results = detector.detectForVideo(current, frameTime); const width = current.videoWidth, height = current.videoHeight;
            if (overlay!.width !== width || overlay!.height !== height) { overlay!.width = width; overlay!.height = height; }
            context!.clearRect(0, 0, width, height);
            const points = results.faceLandmarks[0]; const delta = previousFrameTime ? frameTime - previousFrameTime : 1000 / 30; previousFrameTime = frameTime; const smoothing = 1 - Math.exp(-Math.max(0, delta) / 45);
            if (points) {
              let values = smoothedRef.current;
              if (!values || values.length !== points.length * 3) { values = new Float32Array(points.length * 3); smoothedRef.current = values; for (let i = 0; i < points.length; i++) { const p = i * 3; values[p] = points[i].x; values[p + 1] = points[i].y; values[p + 2] = points[i].z; } }
              else for (let i = 0; i < points.length; i++) { const p = i * 3; values[p] += (points[i].x - values[p]) * smoothing; values[p + 1] += (points[i].y - values[p + 1]) * smoothing; values[p + 2] += (points[i].z - values[p + 2]) * smoothing; }
              drawHints(points, width, height, 1); lostAt = null;
              const leftX = values[33 * 3], leftY = values[33 * 3 + 1];
              const rightX = values[263 * 3], rightY = values[263 * 3 + 1];
              const dx = (rightX - leftX) * width, dy = (rightY - leftY) * height;
              const headPose = estimateHeadRotation(results.facialTransformationMatrixes[0]);
              if (Number.isFinite(leftX + leftY + rightX + rightY)) {
                poseRef.current = { x: values[168 * 3], y: values[168 * 3 + 1],
                    depth: values[168 * 3 + 2],
                    eyeDistance: Math.hypot(dx, dy), videoWidth: width, videoHeight: height,
                    roll: Math.atan2(dy, dx) * 180 / Math.PI, yaw: headPose?.yaw ?? 0, pitch: headPose?.pitch ?? 0,
                      confidence: estimateLandmarkConfidence(points, [
                        33,
                        263,
                        6,
                        168,
                      ]) };
              }
              if (!hadFace) { hadFace = true; setFaceDetected(true); setLandmarkCount(points.length); setStatus("Face tracking active"); }
              if (frameTime - lastMetricsUpdate >= 200) { lastMetricsUpdate = frameTime; setMeasurements(measureFace(points, width, height, results.facialTransformationMatrixes[0])); }
            } else {
              if (hadFace) { hadFace = false; lostAt = frameTime; poseRef.current = null; setFaceDetected(false); setLandmarkCount(0); setMeasurements(null); setStatus("Looking for your face…"); }
              const timeLost = lostAt === null ? Infinity : frameTime - lostAt;
              if (timeLost < 350) drawHints([], width, height, 1 - timeLost / 350); else if (timeLost >= 350) smoothedRef.current = null;
            }
          }
          animationRef.current = requestAnimationFrame(detect);
        } catch (trackingError) {
          console.error("Face tracking stopped:", trackingError); animationRef.current = null; poseRef.current = null;
          setError(trackingError instanceof Error ? trackingError.message : "Face tracking stopped unexpectedly."); setStatus("Tracking stopped"); setFaceDetected(false); setLandmarkCount(0); setMeasurements(null);
        }
      }
      animationRef.current = requestAnimationFrame(detect);
    } catch (startError) {
      if (operation !== operationRef.current) return;
      console.error("Unable to start camera and face tracking:", startError); releaseCameraResources(); setCameraActive(false); setMode(null); setError(describeCameraError(startError)); setStatus("Unable to start camera");
    } finally { if (operation === operationRef.current) { startingRef.current = false; setStarting(false); } }
  }

  function stopCamera() {
    operationRef.current++; startingRef.current = false; poseRef.current = null; faceMeshIndicesRef.current = null; releaseCameraResources(); setStarting(false); setCameraActive(false); setFaceDetected(false); setLandmarkCount(0); setMode(null); setMeasurements(null); setError(""); setStatus("Ready when you are");
  }
  useEffect(() => () => { operationRef.current++; startingRef.current = false; releaseCameraResources(); }, []);
  useEffect(() => {
    if (!modelUrl.startsWith("blob:")) return;
    return () => URL.revokeObjectURL(modelUrl);
  }, [modelUrl]);
  useEffect(() => {
    if (!uploadedImage?.startsWith("blob:")) return;
    return () => URL.revokeObjectURL(uploadedImage);
  }, [uploadedImage]);
  useEffect(() => {
    if (!fullScreen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setFullScreen(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [fullScreen]);

  function resetCalibration() {
    setScale(1);
    setOffsetY(0);
    setDepthOffset(0);
    setRotationOffset({ pitch: 0, yaw: 0, roll: 0 });
  }

  async function uploadFile(file?: File) {
    if (!file) return;
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
      setError("Choose a PNG, JPG, or WebP image.");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setError("Image files must be smaller than 15 MB.");
      return;
    }
    try {
      const bitmap = await createImageBitmap(file);
      const validDimensions = bitmap.width <= 8192 && bitmap.height <= 8192;
      bitmap.close();
      if (!validDimensions) {
        setError("Image dimensions must be 8192 × 8192 pixels or smaller.");
        return;
      }
      resetCalibration();
      setUploadedImage(URL.createObjectURL(file));
      setShowUploadControls(true);
      setError("");
    } catch (imageError) {
      console.error("Unable to read uploaded glasses image:", imageError);
      setError("That image could not be read. Try another PNG, JPG, or WebP.");
    }
  }
  async function uploadModel(file?: File) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".glb")) { setError("Choose a .glb model with its textures embedded."); return; }
    if (file.size > 50 * 1024 * 1024) { setError("3D models must be smaller than 50 MB."); return; }
    try {
      const header = new DataView(await file.slice(0, 12).arrayBuffer());
      if (
        file.size < 12 ||
        header.getUint32(0, true) !== 0x46546c67 ||
        header.getUint32(4, true) !== 2 ||
        header.getUint32(8, true) !== file.size
      ) {
        setError("This is not a valid binary glTF 2.0 (.glb) file.");
        return;
      }
    } catch (modelError) {
      console.error("Unable to validate uploaded GLB:", modelError);
      setError("That model could not be read. Choose a valid GLB 2.0 file.");
      return;
    }
    resetCalibration();
    setUploadedImage(null);
    setModelUrl(URL.createObjectURL(file)); setError(""); setShowUploadControls(true);
  }
  function capture() {
    const video = videoRef.current; if (!video || !video.videoWidth) return;
    const shot = document.createElement("canvas"); shot.width = video.videoWidth; shot.height = video.videoHeight;
    const ctx = shot.getContext("2d"); if (!ctx) return;
    ctx.save(); ctx.translate(shot.width, 0); ctx.scale(-1, 1); ctx.drawImage(video, 0, 0);
    if (canvasRef.current) ctx.drawImage(canvasRef.current, 0, 0, shot.width, shot.height);
    ctx.restore();
    const glassesCanvas = document.querySelector<HTMLCanvasElement>(".glasses-3d-layer canvas");
    if (glassesCanvas) ctx.drawImage(glassesCanvas, 0, 0, shot.width, shot.height);
    setCaptured(shot.toDataURL("image/jpeg", .9));
  }
  const faceShape = getFaceShape(measurements);
  const frame = FRAMES[selected];
  const customModel = modelUrl.startsWith("blob:");

  return (
    <main className={fullScreen ? "studio fullscreen" : "studio"}>
      {!fullScreen && <>
        <div className="announcement"><span>Complimentary worldwide shipping</span><span className="announce-divider">✦</span><span>Thoughtful design. Exceptional fit.</span></div>
        <header className="site-header">
          <a className="wordmark" href="#top" aria-label="Forma home">forma<span>®</span></a>
          <nav><a href="#collection">Eyewear</a><a href="#try-on">Virtual try-on</a><a href="#about">Our approach</a></nav>
          <button className="bag-button" aria-label="Shopping bag">Bag <span>0</span></button>
        </header>
      </>}
      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow"><span className="eyebrow-line"/> PERSONAL FIT, REIMAGINED</p>
          <h1>Find the frame<br/>that feels <em>like you.</em></h1>
          <p className="hero-intro">Thoughtful shapes, considered materials, and a little help finding the one that belongs on your face.</p>
          <a className="text-link" href="#try-on">Discover your fit <span>↗</span></a>
          <div className="hero-note"><span className="note-star">✳</span><span>Made for your face.<br/>Chosen by you.</span></div>
        </div>
        <div className="hero-art" aria-label="Illustration of sculptural eyewear">
          <div className="sun-disc"/><div className="hero-orbit orbit-one"/><div className="hero-orbit orbit-two"/>
          <svg className="hero-glasses" viewBox="0 0 660 300" role="img" aria-label="Sculptural amber eyewear illustration">
            <defs><linearGradient id="acetate" x1="0" y1="0" x2="0.9" y2="1"><stop stopColor="#503320"/><stop offset=".45" stopColor="#ad7646"/><stop offset="1" stopColor="#382319"/></linearGradient><linearGradient id="lens" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#f9ead7" stopOpacity=".17"/><stop offset="1" stopColor="#a87047" stopOpacity=".42"/></linearGradient></defs>
            <path d="M68 84 Q95 54 174 64 L282 82 Q303 88 303 111 L291 207 Q287 243 253 249 L133 257 Q96 259 82 225 L48 134 Q36 104 68 84Z" fill="url(#lens)" stroke="url(#acetate)" strokeWidth="21" strokeLinejoin="round"/>
            <path d="M352 88 Q369 76 396 80 L570 98 Q607 103 597 138 L567 220 Q554 254 516 249 L399 236 Q367 232 362 202 L340 112 Q337 97 352 88Z" fill="url(#lens)" stroke="url(#acetate)" strokeWidth="21" strokeLinejoin="round"/>
            <path d="M296 96 Q325 76 349 96" fill="none" stroke="#6e472d" strokeWidth="19" strokeLinecap="round"/><path d="M54 109 L24 99 Q10 95 4 104" fill="none" stroke="#513421" strokeWidth="17" strokeLinecap="round"/><path d="M602 121 L639 110 Q650 108 656 116" fill="none" stroke="#513421" strokeWidth="17" strokeLinecap="round"/>
          </svg>
          <div className="hero-caption"><span>01 / OBJECTS WITH INTENTION</span><span>EST. IN EVERY DETAIL</span></div>
          <div className="art-index">F—01</div>
        </div>
        <div className="hero-bottom"><span>INDEPENDENT EYEWEAR, CONSIDERED</span><span className="down-arrow">↓</span><span>SCROLL TO EXPLORE</span></div>
      </section>

      <section className="try-section" id="try-on">
        <div className="section-heading"><div><p className="eyebrow"><span className="eyebrow-line"/> THE FITTING ROOM</p><h2>See yourself<br/><em>in a new light.</em></h2></div><p className="section-aside">Allow camera access, then look into the camera. Your selected frame will follow your eye line.</p></div>
        <div className="try-grid">
          <div className="camera-column">
            <div className="camera-frame">
              <div className="camera-topline"><span><i className={cameraActive ? "live-dot active" : "live-dot"}/> {cameraActive ? "LIVE FITTING" : "YOUR PRIVATE FITTING ROOM"}</span><span>01 — 04</span></div>
              <div className="camera-view" id="camera-view">
                <video ref={videoRef} autoPlay playsInline muted className="camera-video"/>
                <canvas ref={canvasRef} className="tracking-canvas"/>
                <Glasses3D poseRef={poseRef} landmarksRef={smoothedRef} faceMeshIndicesRef={faceMeshIndicesRef} modelUrl={modelUrl} imageUrl={uploadedImage} cameraActive={cameraActive} scale={scale} offsetY={offsetY} depthOffset={depthOffset} rotationOffset={rotationOffset} onModelStatus={setModelStatus}/>
                {cameraActive && faceDetected && <div className="face-label"><span className="face-pulse"/> FACE TRACKING <span>·</span> HEAD POSE ACTIVE</div>}
                {!cameraActive && <div className="camera-placeholder"><div className="portrait"><div className="portrait-head"><span className="portrait-hair"/><span className="portrait-neck"/></div><div className="portrait-shoulder"/></div><div className="placeholder-copy"><span className="placeholder-icon">✳</span><span>Make this space yours.</span><small>Your camera stays on this device.</small></div></div>}
                {starting && <div className="camera-loading"><span className="spinner"/> Preparing your fitting room</div>}
                <div className="camera-corners"><i/><i/><i/><i/></div>
                <button className="expand-button" onClick={() => setFullScreen(!fullScreen)} aria-label={fullScreen ? "Exit full screen" : "Enter full screen"} aria-pressed={fullScreen}>{fullScreen ? "↙" : "⛶"}</button>
              </div>
              <div className="camera-controls"><div className="control-status"><span className={faceDetected ? "status-orb detected" : "status-orb"}/><div><b>{faceDetected ? "You're in view" : status}</b><small>{faceDetected ? `${landmarkCount} facial landmarks · ${mode ?? ""} tracking` : "Center your face in the frame"}</small></div></div>
                <div className="control-actions">{cameraActive ? <><button className="capture-button" onClick={capture} aria-label="Capture photograph">◎</button><button className="camera-button stop-button" onClick={stopCamera}>End session <span>×</span></button></> : <button className="camera-button" onClick={startCamera} disabled={starting}>{starting ? "Preparing…" : "Start your fitting"}<span>↗</span></button>}</div>
              </div>
            </div>
            {error && <p role="alert" className="error-message">{error}</p>}
            <div className="privacy-note"><span>◈</span> Your camera is processed on this device. No images are uploaded or stored.</div>
          </div>
          <aside className="selection-panel">
            <div className="selection-top"><div><p className="eyebrow">YOUR FRAME</p><h3>{uploadedImage ? "2D reference image" : customModel ? "Uploaded GLB model" : frame.name}</h3><p className="frame-style">{uploadedImage ? "Frontal image · no hidden depth" : customModel ? "Custom 3D model · calibrate fit" : frame.style}</p></div><span className="frame-price">{uploadedImage || customModel ? "—" : frame.price}</span></div>
            <div className="selected-swatch">{!uploadedImage && !customModel && frame.image ? <div className="real-frame-swatch" style={{ backgroundImage: `url("${frame.image}")` }} role="img" aria-label={frame.name}/> : <div className="swatch-art" style={{ "--frame-tone": uploadedImage ? "#9d795b" : frame.color } as React.CSSProperties}><span/><i/><b/><em/></div>}<span>{uploadedImage ? "CUSTOM IMAGE" : customModel ? "CUSTOM GLB" : frame.shape.toUpperCase()} <span>·</span> {customModel || uploadedImage ? "LOCAL PREVIEW" : "HAND-FINISHED"}</span></div>
            <div className="panel-rule"/>
            <div className="choose-title"><span>EXPLORE THE COLLECTION</span><span>0{selected + 1} / 04</span></div>
            <div className="frame-options">{FRAMES.map((item, i) => <button key={item.name} className={`frame-option ${selected === i && !uploadedImage ? "selected" : ""}`} onClick={() => { setSelected(i); setUploadedImage(null); setModelUrl(DEFAULT_MODEL_URL); resetCalibration(); }} aria-label={`Select ${item.name}`}><span className="option-number">0{i + 1}</span><span className="option-mini" style={{ "--frame-tone": item.color } as React.CSSProperties}><i/><b/></span><span className="option-name">{item.name}</span><span className="option-check">{selected === i && !uploadedImage ? "✓" : "↗"}</span></button>)}</div>
            <div className="upload-row"><button onClick={() => fileRef.current?.click()}><span>＋</span> Add PNG / image</button><input ref={fileRef} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={(event) => { void uploadFile(event.target.files?.[0]); event.currentTarget.value = ""; }}/><button onClick={() => modelFileRef.current?.click()}><span>＋</span> Add GLB model</button><input ref={modelFileRef} type="file" accept=".glb,model/gltf-binary" hidden onChange={(event) => { void uploadModel(event.target.files?.[0]); event.currentTarget.value = ""; }}/>{uploadedImage && <button className="clear-upload" onClick={() => { setUploadedImage(null); setModelUrl(DEFAULT_MODEL_URL); resetCalibration(); }}>Clear ×</button>}<button className="calibrate-toggle" onClick={() => setShowUploadControls(!showUploadControls)}>Adjust</button></div>
            {showUploadControls && <div className="calibration"><label>Scale <input type="range" min=".6" max="1.5" step=".01" value={scale} onChange={(e) => setScale(Number(e.target.value))}/></label><label>Vertical position <input type="range" min="-50" max="50" value={offsetY} onChange={(e) => setOffsetY(Number(e.target.value))}/></label>{!uploadedImage && <><label>Bridge depth <input type="range" min="-80" max="80" value={depthOffset} onChange={(e) => setDepthOffset(Number(e.target.value))}/></label><label>Yaw alignment <input type="range" min="-180" max="180" value={rotationOffset.yaw} onChange={(e) => setRotationOffset({ ...rotationOffset, yaw: Number(e.target.value) })}/></label><label>Pitch alignment <input type="range" min="-90" max="90" value={rotationOffset.pitch} onChange={(e) => setRotationOffset({ ...rotationOffset, pitch: Number(e.target.value) })}/></label></>}<label>Roll alignment <input type="range" min="-45" max="45" value={rotationOffset.roll} onChange={(e) => setRotationOffset({ ...rotationOffset, roll: Number(e.target.value) })}/></label><p>{uploadedImage ? "2D alignment only · image depth is unavailable" : "3D calibration · normalized model width and bridge depth"}</p></div>}
            {uploadedImage && <p className="asset-limit-note">A flat image can only approximate a frontal view. It fades as you turn; it has no real depth, temples, or side geometry. Use a GLB model for multi-angle fitting.</p>}
            {!uploadedImage && !customModel && selected > 0 && <p className="asset-limit-note">The included starter GLB is used for this fitting preview until a product-specific model is supplied.</p>}
            <p className="model-status">◈ {modelStatus}</p>
            <div className="analysis-card"><div className="analysis-title"><span>YOUR STYLE NOTES</span><button onClick={() => setShowAnalysis(!showAnalysis)}>{showAnalysis ? "−" : "+"}</button></div>{showAnalysis && <><div className="analysis-shape"><span className="shape-icon">◯</span><div><small>ESTIMATED FACE SHAPE</small><b>{faceShape.label}</b></div><span className="confidence">STYLE GUIDE</span></div><p className="recommendation">{faceShape.copy}</p>{measurements && <div className="ratio-row"><span>FACE WIDTH / HEIGHT</span><b>{measurements.faceHeight > 0 ? (measurements.faceWidth / measurements.faceHeight).toFixed(2) : "—"}</b></div>}<p className="approx-note">Approximate style guidance from facial proportions, not a biometric or medical assessment.</p></>}</div>
            <div className="fit-tip"><span>✳</span><p><b>A good fit should feel effortless.</b><br/>We’ll help you find your shape, your way.</p></div>
          </aside>
        </div>
      </section>

      <section className="collection-section" id="collection"><div className="collection-head"><div><p className="eyebrow"><span className="eyebrow-line"/> MADE TO BE YOURS</p><h2>Considered shapes.<br/><em>Everyday icons.</em></h2></div><a href="#try-on" className="text-link">Explore all frames <span>↗</span></a></div><div className="product-grid">{FRAMES.slice(0, 3).map((item, index) => <button className="product-card" key={item.name} onClick={() => { setSelected(index); setUploadedImage(null); setModelUrl(DEFAULT_MODEL_URL); resetCalibration(); document.getElementById("try-on")?.scrollIntoView({ behavior: "smooth" }); }}><span className={`product-art art-${index} ${item.image ? "product-with-image" : ""}`}>{item.image ? <span className="product-image" style={{ backgroundImage: `url("${item.image}")` }} role="img" aria-label={item.name}/> : <><span className="product-lens left"/><span className="product-lens right"/><span className="product-bridge"/><span className="product-temple left"/><span className="product-temple right"/></>}<span className="product-number">0{index + 1}</span><span className="product-annotation">FORMA<br/>STUDIO</span></span><span className="product-meta"><span><b>{item.name}</b><small>{item.style}</small></span><b>{item.price}</b></span></button>)}</div></section>
      <section className="manifesto" id="about"><div className="manifesto-mark">f.</div><p>Good design should feel<br/>like <em>it was always yours.</em></p><span>DESIGNED WITH INTENTION · WORN YOUR WAY</span></section>
      <footer className="site-footer"><a className="wordmark" href="#top">forma<span>®</span></a><span>OBJECTS FOR THE WAY YOU SEE.</span><span>© FORMA STUDIO 2025</span></footer>
      {captured && <div className="capture-modal" role="dialog" aria-modal="true" aria-label="Your fitting photo"><div className="capture-card"><button className="modal-close" onClick={() => setCaptured(null)}>×</button><div className="capture-image" role="img" aria-label="Your fitting room photo" style={{ backgroundImage: `url("${captured}")` }}/><div><b>Your moment, your device.</b><span>This preview has not been uploaded.</span><a href={captured} download="forma-fitting.jpg">Save photo ↗</a></div></div></div>}
    </main>
  );
}

function getFaceShape(m: FaceMeasurements | null) {
  if (!m || !m.faceHeight || !m.faceWidth) return { label: "Waiting for you", copy: "Start your fitting to see gentle style suggestions based on your facial proportions." };
  const ratio = m.faceWidth / m.faceHeight;
  if (ratio > .82) return { label: "Round", copy: "Angular or softly squared frames can add a little definition and balance." };
  if (ratio < .68) return { label: "Oblong", copy: "Wider frames and a deeper lens can bring a lovely sense of balance." };
  if (ratio > .74) return { label: "Square", copy: "Rounded shapes and gentle curves can soften strong, balanced proportions." };
  return { label: "Oval", copy: "Your balanced proportions work beautifully with a wide range of shapes." };
}
function describeCameraError(error: unknown) {
  if (error instanceof Error) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") return "Camera permission was denied. Allow access in your browser settings, then try again.";
    if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") return "No camera was found. Connect a camera and try again.";
    if (error.name === "NotReadableError" || error.name === "TrackStartError") return "The camera could not start. It may be in use by another app.";
    return error.message;
  }
  return "Unable to start the camera.";
}
