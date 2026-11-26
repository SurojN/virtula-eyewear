"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import {
  calibrateModelBounds,
  estimateFrameScale,
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
  imageUrl: string | null;
  cameraActive: boolean;
  scale: number;
  offsetY: number;
  depthOffset: number;
  rotationOffset: { pitch: number; yaw: number; roll: number };
  onModelStatus: (message: string) => void;
};

function lensShape(cx: number, cy: number, inset = 0) {
  const shape = new THREE.Shape();
  const w = 0.78 - inset;
  const h = 0.56 - inset;
  shape.moveTo(cx - w * 0.84, cy + h * 0.78);
  shape.quadraticCurveTo(cx - w * 0.35, cy + h * 1.06, cx + w * 0.76, cy + h * 0.84);
  shape.quadraticCurveTo(cx + w * 1.04, cy + h * 0.8, cx + w, cy + h * 0.35);
  shape.lineTo(cx + w * 0.82, cy - h * 0.63);
  shape.quadraticCurveTo(cx + w * 0.72, cy - h, cx + w * 0.18, cy - h);
  shape.lineTo(cx - w * 0.64, cy - h * 0.94);
  shape.quadraticCurveTo(cx - w, cy - h * 0.88, cx - w * 0.98, cy - h * 0.4);
  shape.lineTo(cx - w * 0.9, cy + h * 0.48);
  shape.quadraticCurveTo(cx - w * 0.9, cy + h * 0.72, cx - w * 0.84, cy + h * 0.78);
  return shape;
}

function tube(points: THREE.Vector3[], radius: number, material: THREE.Material) {
  const curve = new THREE.CatmullRomCurve3(points);
  return new THREE.Mesh(new THREE.TubeGeometry(curve, 28, radius, 10, false), material);
}

function makeProceduralFrame() {
  const root = new THREE.Group();
  const acetate = new THREE.MeshPhysicalMaterial({
    color: 0x151515,
    roughness: 0.22,
    metalness: 0.12,
    clearcoat: 0.78,
    clearcoatRoughness: 0.18,
  });
  const lensMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xb2c8c2,
    transparent: true,
    opacity: 0.17,
    roughness: 0.12,
    metalness: 0.02,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const silver = new THREE.MeshStandardMaterial({
    color: 0xe0ded9,
    metalness: 0.86,
    roughness: 0.22,
  });

  for (const center of [-0.86, 0.86]) {
    const frame = lensShape(center, 0);
    frame.holes.push(lensShape(center, 0, 0.105));
    const frameMesh = new THREE.Mesh(
      new THREE.ExtrudeGeometry(frame, {
        depth: 0.14,
        bevelEnabled: true,
        bevelSegments: 3,
        bevelSize: 0.025,
        bevelThickness: 0.025,
        curveSegments: 12,
      }),
      acetate,
    );
    wrapGeometryAroundFace(frameMesh.geometry);
    frameMesh.position.z = -0.07;
    root.add(frameMesh);
    const lensGeometry = new THREE.ShapeGeometry(lensShape(center, 0, 0.13), 16);
    wrapGeometryAroundFace(lensGeometry);
    const lens = new THREE.Mesh(lensGeometry, lensMaterial);
    lens.position.z = -0.09;
    root.add(lens);

    const hinge = new THREE.Mesh(new THREE.SphereGeometry(0.075, 16, 12), silver);
    hinge.position.set(center + Math.sign(center) * 0.94, 0.14, -0.02);
    hinge.scale.set(1.35, 0.68, 0.48);
    root.add(hinge);
  }

  root.add(
    tube(
      [
        new THREE.Vector3(-0.18, 0.42, 0.08),
        new THREE.Vector3(-0.12, 0.55, 0.12),
        new THREE.Vector3(0, 0.59, 0.14),
        new THREE.Vector3(0.12, 0.55, 0.12),
        new THREE.Vector3(0.18, 0.42, 0.08),
      ],
      0.105,
      acetate,
    ),
  );
  for (const side of [-1, 1]) {
    const temple = tube(
        [
          new THREE.Vector3(side * 1.78, 0.12, -0.035),
          new THREE.Vector3(side * 1.86, 0.1, -0.3),
          new THREE.Vector3(side * 1.88, 0.04, -0.72),
          new THREE.Vector3(side * 1.83, -0.03, -1.13),
          new THREE.Vector3(side * 1.72, -0.1, -1.32),
        ],
        0.09,
        acetate,
      );
    temple.name = side < 0 ? "left_temple_front" : "right_temple_front";
    root.add(temple);
  }
  return root;
}

function wrapGeometryAroundFace(geometry: THREE.BufferGeometry) {
  const position = geometry.getAttribute("position");
  for (let index = 0; index < position.count; index += 1) {
    const x = position.getX(index);
    const z = position.getZ(index);
    // The bridge sits closer to the nose; the outer rims sweep back toward the hinges.
    position.setZ(index, z + 0.12 * Math.max(0, 1 - Math.abs(x) / 1.85));
  }
  position.needsUpdate = true;
  geometry.computeVertexNormals();
}

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

function extendStarterTemples(model: THREE.Group) {
  model.traverse((child) => {
    if (!(child instanceof THREE.Mesh) || !/temple_(front|back|joint)/i.test(child.name)) return;
    // The included frame's arms end too close to the hinge. Extend them back
    // around the head so the face depth mesh can hide them through the lenses.
    child.geometry = child.geometry.clone();
    const position = child.geometry.getAttribute("position");
    for (let index = 0; index < position.count; index += 1) {
      position.setZ(index, position.getZ(index) * 1.5);
    }
    position.needsUpdate = true;
    child.geometry.computeVertexNormals();
    child.geometry.computeBoundingSphere();
  });
}

function prepareTempleVisibility(model: THREE.Group) {
  const temples: Array<{ mesh: THREE.Mesh; side: "left" | "right" }> = [];
  model.traverse((child) => {
    if (!(child instanceof THREE.Mesh) || !/^(left|right)_temple/i.test(child.name)) return;
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

function updateTempleVisibility(
  temples: Array<{ mesh: THREE.Mesh; side: "left" | "right" }>,
  yaw: number,
) {
  // Temples sit behind the clear lenses. Keep them out of the frontal eye area,
  // then reveal the near arm as the wearer turns toward a three-quarter view.
  const turn = THREE.MathUtils.clamp((Math.abs(yaw) - 9) / 27, 0, 1);
  const nearSide = yaw >= 0 ? "right" : "left";
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
  imageUrl,
  cameraActive,
  scale,
  offsetY,
  depthOffset,
  rotationOffset,
  onModelStatus,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const statusRef = useRef(onModelStatus);
  const adjustmentRef = useRef({ scale, offsetY, depthOffset, rotationOffset });
  useEffect(() => {
    statusRef.current = onModelStatus;
  }, [onModelStatus]);
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
    const fallback = makeProceduralFrame();
    glassesRoot.add(fallback);

    let model: THREE.Group | null = null;
    let imagePlane: THREE.Mesh | null = null;
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

    async function loadImage() {
      if (!imageUrl) return;
      statusRef.current("Preparing 2D reference · frontal view only");
      try {
        const texture = await new THREE.TextureLoader().loadAsync(imageUrl);
        if (disposed) {
          texture.dispose();
          return;
        }
        texture.colorSpace = THREE.SRGBColorSpace;
        const aspect = texture.image.width / texture.image.height;
        if (!Number.isFinite(aspect) || aspect <= 0) {
          texture.dispose();
          throw new Error("Image dimensions are invalid.");
        }
        const material = new THREE.MeshBasicMaterial({
          map: texture,
          transparent: true,
          opacity: 1,
          depthTest: true,
          depthWrite: false,
          side: THREE.DoubleSide,
          toneMapped: false,
        });
        imagePlane = new THREE.Mesh(
          new THREE.PlaneGeometry(3.55, 3.55 / aspect),
          material,
        );
        glassesRoot.remove(fallback);
        glassesRoot.add(imagePlane);
        statusRef.current("2D image ready · frontal view only");
      } catch (error) {
        if (!disposed) {
          statusRef.current(
            error instanceof Error
              ? `Image load failed: ${error.message}`
              : "Could not load this image.",
          );
        }
      }
    }

    async function loadModel() {
      if (imageUrl) return;
      if (!modelUrl) {
        statusRef.current("Preparing the included 3D frame…");
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
        if (modelUrl.endsWith("starter-wayfarer-glasses.glb")) {
          extendStarterTemples(model);
        }
        templeMeshes = prepareTempleVisibility(model);
        const bounds = new THREE.Box3().setFromObject(model);
        const calibration = calibrateModelBounds({
          min: bounds.min,
          max: bounds.max,
        });
        if (!calibration) {
          throw new Error("Model has invalid or empty dimensions.");
        }
        model.scale.setScalar(calibration.scale);
        model.position.set(
          calibration.position.x,
          calibration.position.y,
          calibration.position.z,
        );
        glassesRoot.remove(fallback);
        glassesRoot.add(model);
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
    void loadImage();
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
      const pose = poseRef.current;
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
          estimateFrameScale(eyePx, pose.yaw, settings.scale) ?? eyePx / 2;
        targetScale.setScalar(frameScale);
        targetEuler.set(
          THREE.MathUtils.degToRad(pose.pitch + settings.rotationOffset.pitch),
          THREE.MathUtils.degToRad(-pose.yaw + settings.rotationOffset.yaw),
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
        glassesRoot.visible = true;
        renderer.domElement.style.opacity = String(
            THREE.MathUtils.lerp(
              0.72,
              1,
              THREE.MathUtils.clamp(pose.confidence, 0, 1),
            ),
          );

        if (imagePlane) {
          const imageMaterial = imagePlane.material as THREE.MeshBasicMaterial;
          const yaw = Math.abs(pose.yaw);
          imageMaterial.opacity =
            yaw <= 20 ? 1 : yaw >= 55 ? 0 : 1 - (yaw - 20) / 35;
        }
      } else {
        faceOccluder.visible = false;
        if (!cameraActive || !hasTrackedPose) {
          missingPoseAt = null;
          hasPose = false;
          glassesRoot.position.set(0, 0, 0);
          glassesRoot.rotation.set(0, 0, 0);
          glassesRoot.scale.setScalar(
            (Math.min(width, height * 1.35) * 0.55) / 3.55,
          );
          glassesRoot.visible = true;
          renderer.domElement.style.opacity = "1";
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
      if (imagePlane) disposeObject(imagePlane);
      disposeObject(fallback);
      disposeObject(faceOccluder);
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [
    faceMeshIndicesRef,
    imageUrl,
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
