export type ModelBounds = {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
};

export type ModelCalibration = {
  scale: number;
  position: { x: number; y: number; z: number };
};

export type ModelAnchor = { x: number; y: number; z: number };

export type ScanScaleProfiles = {
  front?: { yaw: number; eyeDistance: number };
  left?: { yaw: number; eyeDistance: number };
  right?: { yaw: number; eyeDistance: number };
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

export function estimateScannedFrameScale(
  eyeDistancePixels: number,
  yawDegrees: number,
  fitScale: number,
  profiles: ScanScaleProfiles,
) {
  const front = profiles.front;
  const side = yawDegrees >= 0 ? profiles.left : profiles.right;
  if (
    !front ||
    !side ||
    ![front.eyeDistance, front.yaw, side.eyeDistance, side.yaw].every(Number.isFinite) ||
    front.eyeDistance <= 0 ||
    side.eyeDistance <= 0 ||
    Math.abs(side.yaw) < 20
  ) {
    return estimateFrameScale(eyeDistancePixels, yawDegrees, fitScale);
  }

  const angleFraction = Math.min(1, Math.abs(yawDegrees / side.yaw));
  const expectedEyeDistance =
    front.eyeDistance + (side.eyeDistance - front.eyeDistance) * angleFraction;
  if (expectedEyeDistance <= 0) {
    return estimateFrameScale(eyeDistancePixels, yawDegrees, fitScale);
  }

  // The profile scan supplies the wearer's actual projected eye spacing at
  // each angle. Interpolating it avoids relying on a generic cosine-only fit.
  return ((front.eyeDistance / 2) * eyeDistancePixels / expectedEyeDistance) * fitScale;
}

export function calibrateModelBounds(
  bounds: ModelBounds,
  targetWidth = 3.55,
  anchor?: ModelAnchor | null,
): ModelCalibration | null {
  const width = bounds.max.x - bounds.min.x;
  const height = bounds.max.y - bounds.min.y;
  const depth = bounds.max.z - bounds.min.z;
  if (
    ![width, height, depth, targetWidth].every(Number.isFinite) ||
    width <= 0 ||
    height <= 0 ||
    depth < 0 ||
    targetWidth <= 0 ||
    (anchor && ![anchor.x, anchor.y, anchor.z].every(Number.isFinite))
  ) {
    return null;
  }

  const scale = targetWidth / width;
  const centerX = anchor?.x ?? (bounds.min.x + bounds.max.x) / 2;
  const centerY = anchor?.y ?? (bounds.min.y + bounds.max.y) / 2;
  const frontZ = anchor?.z ?? bounds.max.z;
  return {
    scale,
    position: {
      x: centerX === 0 ? 0 : -centerX * scale,
      y: centerY === 0 ? 0 : -centerY * scale,
      z: -frontZ * scale,
    },
  };
}
