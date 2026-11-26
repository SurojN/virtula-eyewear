"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import {
  calibrateModelBounds,
  estimateScannedFrameScale,
  type ScanScaleProfiles,
} from "@/lib/glasses-calibration";
import { projectFaceDepth, projectMirroredPoint } from "@/lib/rendering";

export type FacePose = {
  x: number;
  y: number;
  depth: number;
  eyeDistance: number;
  videoWidth: number;
  videoHeight: number;
  roll: number;
  yaw: number;
  pitch: number;
  confidence: number;
} | null;

type Props = {
  poseRef: { current: FacePose };
  landmarksRef: { current: Float32Array | null };
  faceMeshIndicesRef: { current: Uint16Array | null };
  modelUrl: string;
  scanProfiles: ScanScaleProfiles;
  cameraActive: boolean;
  scanComplete: boolean;
  scale: number;
  offsetY: number;
  depthOffset: number;
  rotationOffset: { pitch: number; yaw: number; roll: number };
  onModelStatus: (message: string) => void;
};

function disposeObject(root: THREE.Object3D) {
  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    child.geometry.dispose();
    const materials = Array.isArray(child.material)
      ? child.material
      : [child.material];
    for (const material of materials) {
      for (const value of Object.values(material)) {
        if (value instanceof THREE.Texture) value.dispose();
      }
      material.dispose();
    }
  });
}

function buildFaceOccluder() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(478 * 3), 3).setUsage(
      THREE.DynamicDrawUsage,
    ),
  );
  const material = new THREE.MeshBasicMaterial({
    colorWrite: false,
    depthTest: true,
    depthWrite: true,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = -1;
  return mesh;
}

function prepareTempleVisibility(model: THREE.Group) {
  const temples: Array<{ mesh: THREE.Mesh; side: "left" | "right" }> = [];
  model.traverse((child) => {
    if (!(child instanceof THREE.Mesh) || !/^(left|right)_.*(?:temple|ear_tip)/i.test(child.name)) return;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    child.material = materials.map((source) => {
      const material = source.clone();
      material.transparent = true;
      material.depthWrite = false;
      material.opacity = 0;
      return material;
    });
    child.userData.templeSide = child.name.toLowerCase().startsWith("left_") ? "left" : "right";
    temples.push({ mesh: child, side: child.userData.templeSide });
  });
  return temples;
}

function findBridgeAnchor(model: THREE.Group, modelBounds: THREE.Box3) {
  const modelWidth = modelBounds.max.x - modelBounds.min.x;
  const candidates: Array<{ anchor: THREE.Vector3; score: number }> = [];
  model.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const name = child.name.toLowerCase();
    if (!/(nose|bridge)/.test(name) || /(pad|support|arm)/.test(name)) return;
    const score = name === "bridge"
      ? 10
      : name.includes("nose")
        ? 8
        : name.includes("bridge")
          ? 7
          : 0;
    if (score === 0) return;
    const partBounds = new THREE.Box3().setFromObject(child);
    const partSize = partBounds.getSize(new THREE.Vector3());
    if (!Number.isFinite(partSize.x) || partSize.x > modelWidth * 0.6) return;
    candidates.push({ anchor: partBounds.getCenter(new THREE.Vector3()), score });
  });
  candidates.sort((first, second) => second.score - first.score);
  return candidates[0]?.anchor ?? null;
}

function updateTempleVisibility(
  temples: Array<{ mesh: THREE.Mesh; side: "left" | "right" }>,
  yaw: number,
) {
  // Temples sit behind the clear lenses. Keep them out of the frontal eye area,
  // then reveal the near arm as the wearer turns toward a three-quarter view.
  const turn = THREE.MathUtils.clamp((Math.abs(yaw) - 9) / 27, 0, 1);
  // FacePose yaw is normalized to the mirrored selfie preview: positive yaw
  // means the wearer has turned toward their own left side.
  const nearSide = yaw >= 0 ? "left" : "right";
  for (const { mesh, side } of temples) {
    const opacity = side === nearSide ? turn : turn > 0.82 ? (turn - 0.82) * 0.35 : 0;
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials) material.opacity = opacity;
  }
}

export default function Glasses3D({
  poseRef,
  landmarksRef,
  faceMeshIndicesRef,
  modelUrl,
  scanProfiles,
  cameraActive,
  scanComplete,
  scale,
  offsetY,
  depthOffset,
  rotationOffset,
  onModelStatus,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const statusRef = useRef(onModelStatus);
  const scanProfilesRef = useRef(scanProfiles);
  const scanCompleteRef = useRef(scanComplete);
  const adjustmentRef = useRef({ scale, offsetY, depthOffset, rotationOffset });
  useEffect(() => {
    statusRef.current = onModelStatus;
  }, [onModelStatus]);
  useEffect(() => {
    scanProfilesRef.current = scanProfiles;
  }, [scanProfiles]);
  useEffect(() => {
    scanCompleteRef.current = scanComplete;
  }, [scanComplete]);
  useEffect(() => {
    adjustmentRef.current = { scale, offsetY, depthOffset, rotationOffset };
  }, [scale, offsetY, depthOffset, rotationOffset]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const renderHost = host;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: true,
        powerPreference: "high-performance",
        preserveDrawingBuffer: true,
      });
    } catch (error) {
      statusRef.current(
        error instanceof Error
          ? `3D renderer unavailable: ${error.message}`
          : "3D renderer unavailable on this device.",
      );
      return;
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(host.clientWidth, host.clientHeight, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.domElement.style.transition = "opacity 120ms linear";
    host.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("aria-hidden", "true");

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      35,
      host.clientWidth / Math.max(1, host.clientHeight),
      0.1,
      10000,
    );
    camera.position.z =
      host.clientHeight / (2 * Math.tan(THREE.MathUtils.degToRad(35) / 2));
    scene.add(new THREE.AmbientLight(0xffffff, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 3.2);
    key.position.set(-140, 180, 360);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xd8e4ff, 1.2);
    fill.position.set(180, 20, 100);
    scene.add(fill);

    const glassesRoot = new THREE.Group();
    glassesRoot.renderOrder = 1;
    scene.add(glassesRoot);
    const faceOccluder = buildFaceOccluder();
    scene.add(faceOccluder);
    glassesRoot.visible = false;
    let hasModel = false;

    let model: THREE.Group | null = null;
    let templeMeshes: Array<{ mesh: THREE.Mesh; side: "left" | "right" }> = [];
    let raf = 0;
    let disposed = false;
    let lastWidth = 0;
    let lastHeight = 0;
    let lastPoseAt = 0;
    let hasPose = false;
    let hasTrackedPose = false;
    let missingPoseAt: number | null = null;
    const targetPosition = new THREE.Vector3();
    const targetScale = new THREE.Vector3();
    const targetEuler = new THREE.Euler(0, 0, 0, "XYZ");
    const targetRotation = new THREE.Quaternion();

    async function loadModel() {
      if (!modelUrl) {
        statusRef.current("Choose a GLB frame model to begin fitting.");
        return;
      }
      statusRef.current("Loading 3D frame model…");
      try {
        const gltf = await new GLTFLoader().loadAsync(modelUrl);
        if (disposed) {
          disposeObject(gltf.scene);
          return;
        }
        model = gltf.scene;
        model.updateMatrixWorld(true);
        templeMeshes = prepareTempleVisibility(model);
        const bounds = new THREE.Box3().setFromObject(model);
        const bridgeAnchor = findBridgeAnchor(model, bounds);
        const calibration = calibrateModelBounds({
          min: bounds.min,
          max: bounds.max,
        }, 3.55, bridgeAnchor);
        if (!calibration) {
          throw new Error("Model has invalid or empty dimensions.");
        }
        model.scale.setScalar(calibration.scale);
        model.position.set(
          calibration.position.x,
          calibration.position.y,
          calibration.position.z,
        );
        glassesRoot.add(model);
        hasModel = true;
        statusRef.current("3D model ready · adjust fit for this frame");
      } catch (error) {
        if (!disposed) {
          statusRef.current(
            error instanceof Error
              ? `Model load failed: ${error.message}`
              : "Could not load this GLB. Use a valid model with embedded textures.",
          );
        }
      }
    }
    void loadModel();

    const resizeObserver = new ResizeObserver(() => {
      if (!host) return;
      lastWidth = host.clientWidth;
      lastHeight = host.clientHeight;
      renderer.setSize(lastWidth, lastHeight, false);
      camera.aspect = lastWidth / Math.max(1, lastHeight);
      camera.position.z =
        lastHeight / (2 * Math.tan(THREE.MathUtils.degToRad(35) / 2));
      camera.updateProjectionMatrix();
    });
    resizeObserver.observe(host);

    function render(frameTime: number) {
      const pose = cameraActive && scanCompleteRef.current ? poseRef.current : null;
      const width = renderHost.clientWidth;
      const height = renderHost.clientHeight;
      if (width !== lastWidth || height !== lastHeight) {
        lastWidth = width;
        lastHeight = height;
        renderer.setSize(width, height, false);
        camera.aspect = width / Math.max(height, 1);
        camera.position.z =
          height / (2 * Math.tan(THREE.MathUtils.degToRad(35) / 2));
        camera.updateProjectionMatrix();
      }

      if (pose && width > 0 && height > 0) {
        hasTrackedPose = true;
        updateTempleVisibility(templeMeshes, pose.yaw);
        const centerProjection = projectMirroredPoint(
          pose.x,
          pose.y,
          pose.videoWidth,
          pose.videoHeight,
          width,
          height,
        );
        if (!centerProjection) {
          renderer.domElement.style.opacity = "0";
          renderer.render(scene, camera);
          raf = requestAnimationFrame(render);
          return;
        }
        const coverScale = centerProjection.coverScale;
        const bridgeDepth = projectFaceDepth(
          pose.depth,
          pose.videoWidth,
          coverScale,
        );
        if (bridgeDepth === null) {
          renderer.domElement.style.opacity = "0";
          renderer.render(scene, camera);
          raf = requestAnimationFrame(render);
          return;
        }
        const renderedWidth = pose.videoWidth * coverScale;
        const renderedHeight = pose.videoHeight * coverScale;
        const cropX = (width - renderedWidth) / 2;
        const cropY = (height - renderedHeight) / 2;
        const eyePx = pose.eyeDistance * coverScale;
        const landmarks = landmarksRef.current;
        const positions = faceOccluder.geometry.getAttribute(
          "position",
        ) as THREE.BufferAttribute;
        const faceIndices = faceMeshIndicesRef.current;
        if (faceIndices?.length && !faceOccluder.geometry.index) {
          faceOccluder.geometry.setIndex(Array.from(faceIndices));
        }

        if (
          landmarks &&
          landmarks.length >= 478 * 3 &&
          faceIndices?.length &&
          faceOccluder.geometry.index
        ) {
          let validFaceMesh = true;
          for (let index = 0; index < 478; index += 1) {
            const source = index * 3;
            const screenX =
              width -
              (landmarks[source] * renderedWidth + cropX) -
              width / 2;
            const y = height / 2 - (landmarks[source + 1] * renderedHeight + cropY);
            const z = projectFaceDepth(
              landmarks[source + 2],
              pose.videoWidth,
              coverScale,
            );
            if (z === null) {
              validFaceMesh = false;
              break;
            }
            positions.setXYZ(index, screenX, y, z);
          }
          positions.needsUpdate = true;
          faceOccluder.visible = validFaceMesh;
        } else {
          faceOccluder.visible = false;
        }
        const settings = adjustmentRef.current;
        targetPosition.set(
          centerProjection.x,
          centerProjection.y + settings.offsetY,
          bridgeDepth + eyePx * 0.12 + settings.depthOffset,
        );
        const frameScale =
          estimateScannedFrameScale(eyePx, pose.yaw, settings.scale, scanProfilesRef.current) ?? eyePx / 2;
        targetScale.setScalar(frameScale);
        targetEuler.set(
          THREE.MathUtils.degToRad(-pose.pitch + settings.rotationOffset.pitch),
          THREE.MathUtils.degToRad(pose.yaw + settings.rotationOffset.yaw),
          THREE.MathUtils.degToRad(-pose.roll + settings.rotationOffset.roll),
          "XYZ",
        );
        targetRotation.setFromEuler(targetEuler);

        const smoothing = hasPose
          ? 1 - Math.exp(-Math.max(1, frameTime - lastPoseAt) / 38)
          : 1;
        if (hasPose) {
          glassesRoot.position.lerp(targetPosition, smoothing);
          glassesRoot.scale.lerp(targetScale, smoothing);
          glassesRoot.quaternion.slerp(targetRotation, smoothing);
        } else {
          glassesRoot.position.copy(targetPosition);
          glassesRoot.scale.copy(targetScale);
          glassesRoot.quaternion.copy(targetRotation);
          hasPose = true;
        }

        lastPoseAt = frameTime;
        missingPoseAt = null;
        glassesRoot.visible = hasModel;
        renderer.domElement.style.opacity = String(
            THREE.MathUtils.lerp(
              0.72,
              1,
              THREE.MathUtils.clamp(pose.confidence, 0, 1),
            ),
          );


      } else {
        faceOccluder.visible = false;
        if (!cameraActive || !scanCompleteRef.current) {
          missingPoseAt = null;
          hasPose = false;
          glassesRoot.visible = false;
          renderer.domElement.style.opacity = "1";
        } else if (!hasTrackedPose) {
          missingPoseAt = null;
          glassesRoot.visible = false;
        } else {
          if (missingPoseAt === null) missingPoseAt = frameTime;
          const timeWithoutPose = frameTime - missingPoseAt;
          renderer.domElement.style.opacity = String(
            Math.max(0, 1 - timeWithoutPose / 350),
          );
          if (timeWithoutPose > 350) {
            glassesRoot.visible = false;
            hasPose = false;
          }
        }
      }

      renderer.render(scene, camera);
      raf = requestAnimationFrame(render);
    }
    raf = requestAnimationFrame(render);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      if (model) disposeObject(model);
      disposeObject(faceOccluder);
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [
    faceMeshIndicesRef,
      landmarksRef,
    modelUrl,
    poseRef,
    cameraActive,
  ]);

  return (
    <div
      ref={hostRef}
      className="glasses-3d-layer"
      aria-label="Three dimensional glasses preview"
    />
  );
}
