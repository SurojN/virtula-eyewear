"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

function disposeModel(root: THREE.Object3D) {
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

export default function GlbProductPreview({
  src,
  label,
}: {
  src: string;
  label: string;
}) {
  const hostRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const previewHost: HTMLSpanElement = host;

    let active = false;
    let disposed = false;
    let generation = 0;
    let renderer: THREE.WebGLRenderer | null = null;
    let scene: THREE.Scene | null = null;
    let camera: THREE.PerspectiveCamera | null = null;
    let model: THREE.Group | null = null;
    let resizeObserver: ResizeObserver | null = null;
    let intersectionObserver: IntersectionObserver | null = null;

    function disposeActivePreview() {
      active = false;
      generation += 1;
      resizeObserver?.disconnect();
      resizeObserver = null;
      if (model) disposeModel(model);
      model = null;
      if (renderer) {
        renderer.forceContextLoss();
        renderer.dispose();
        renderer.domElement.remove();
      }
      renderer = null;
      scene = null;
      camera = null;
      delete previewHost.dataset.previewReady;
    }

    function render() {
      if (!renderer || !scene || !camera) return;
      const width = previewHost.clientWidth;
      const height = previewHost.clientHeight;
      if (!width || !height) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    }

    async function startPreview() {
      if (disposed || active) return;
      active = true;
      const currentGeneration = ++generation;
      delete previewHost.dataset.previewError;
      try {
        renderer = new THREE.WebGLRenderer({
          alpha: true,
          antialias: true,
          powerPreference: "low-power",
        });
      } catch {
        previewHost.dataset.previewError = "true";
        active = false;
        return;
      }

      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.15;
      previewHost.appendChild(renderer.domElement);

      scene = new THREE.Scene();
      scene.add(new THREE.AmbientLight(0xffffff, 2));
      const key = new THREE.DirectionalLight(0xffffff, 3.2);
      key.position.set(-3, 5, 8);
      scene.add(key);
      const fill = new THREE.DirectionalLight(0xdbe8ff, 1.3);
      fill.position.set(4, 1, 3);
      scene.add(fill);
      camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
      camera.position.set(0, 0.05, 6.8);
      camera.lookAt(0, 0, 0);
      resizeObserver = new ResizeObserver(render);
      resizeObserver.observe(previewHost);

      try {
        const gltf = await new GLTFLoader().loadAsync(src);
        if (disposed || !active || currentGeneration !== generation) {
          disposeModel(gltf.scene);
          return;
        }
        model = gltf.scene;
        const bounds = new THREE.Box3().setFromObject(model);
        const size = bounds.getSize(new THREE.Vector3());
        const center = bounds.getCenter(new THREE.Vector3());
        if (!Number.isFinite(size.x) || size.x <= 0) {
          previewHost.dataset.previewError = "true";
          return;
        }
        const normalization = 3.45 / size.x;
        model.scale.setScalar(normalization);
        model.position.set(
          -center.x * normalization,
          -center.y * normalization,
          -center.z * normalization,
        );
        model.rotation.y = -0.18;
        const root = new THREE.Group();
        root.add(model);
        scene?.add(root);
        previewHost.dataset.previewReady = "true";
        render();
      } catch {
        if (!disposed && active && currentGeneration === generation) {
          previewHost.dataset.previewError = "true";
        }
      }
    }

    if ("IntersectionObserver" in window) {
      intersectionObserver = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (entry.isIntersecting) void startPreview();
            else if (active) disposeActivePreview();
          }
        },
        { rootMargin: "180px 0px" },
      );
      intersectionObserver.observe(previewHost);
    } else {
      void startPreview();
    }

    return () => {
      disposed = true;
      intersectionObserver?.disconnect();
      disposeActivePreview();
    };
  }, [src]);

  return (
    <span
      ref={hostRef}
      className="glb-product-preview"
      role="img"
      aria-label={`${label} 3D model preview`}
    >
      <span className="preview-loading" aria-hidden="true">Loading 3D frame…</span>
    </span>
  );
}
