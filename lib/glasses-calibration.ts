export type ModelBounds = {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
};

export type ModelCalibration = {
  scale: number;
  position: { x: number; y: number; z: number };
};

export function estimateFrameScale(
  eyeDistancePixels: number,
  yawDegrees: number,
  fitScale = 1,
) {
  if (
    ![eyeDistancePixels, yawDegrees, fitScale].every(Number.isFinite) ||
    eyeDistancePixels <= 0 ||
    fitScale <= 0
  ) {
    return null;
  }

  const projectedWidth = Math.max(
    0.4,
    Math.abs(Math.cos((yawDegrees * Math.PI) / 180)),
  );
  return (eyeDistancePixels / 2 / projectedWidth) * fitScale;
}

export function calibrateModelBounds(
  bounds: ModelBounds,
  targetWidth = 3.55,
): ModelCalibration | null {
  const width = bounds.max.x - bounds.min.x;
  const height = bounds.max.y - bounds.min.y;
  const depth = bounds.max.z - bounds.min.z;
  if (
    ![width, height, depth, targetWidth].every(Number.isFinite) ||
    width <= 0 ||
    height <= 0 ||
    depth < 0 ||
    targetWidth <= 0
  ) {
    return null;
  }

  const scale = targetWidth / width;
  const centerX = (bounds.min.x + bounds.max.x) / 2;
  const centerY = (bounds.min.y + bounds.max.y) / 2;
  return {
    scale,
    position: {
      x: centerX === 0 ? 0 : -centerX * scale,
      y: centerY === 0 ? 0 : -centerY * scale,
      z: -bounds.max.z * scale,
    },
  };
}
