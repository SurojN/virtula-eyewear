import type {
  Matrix,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision";

export type FaceMeasurements = {
  faceWidth: number;
  faceHeight: number;
  eyeSpacing: number;
  eyeLineAngle: number;
  pitch: number | null;
  yaw: number | null;
  roll: number | null;
};

export function estimateLandmarkConfidence(
  landmarks: NormalizedLandmark[],
  indices: number[],
) {
  const scores = indices
    .map((index) => landmarks[index]?.presence)
    .filter(
      (score): score is number =>
        typeof score === "number" && Number.isFinite(score),
    );
  if (scores.length === 0) return 1;
  return scores.reduce((sum, score) => sum + score, 0) / scores.length;
}

export type FaceMeshConnection = {
  start: number;
  end: number;
};

export function createFaceMeshIndices(
  connections: FaceMeshConnection[],
  landmarkCount = 478,
) {
  const neighbors = Array.from(
    { length: landmarkCount },
    () => new Set<number>(),
  );
  for (const { start, end } of connections) {
    if (
      start >= 0 &&
      end >= 0 &&
      start < landmarkCount &&
      end < landmarkCount &&
      start !== end
    ) {
      neighbors[start].add(end);
      neighbors[end].add(start);
    }
  }

  const triangles: number[] = [];
  for (let first = 0; first < landmarkCount; first += 1) {
    for (const second of neighbors[first]) {
      if (second <= first) continue;
      for (const third of neighbors[first]) {
        if (third > second && neighbors[second].has(third)) {
          triangles.push(first, second, third);
        }
      }
    }
  }
  return Uint16Array.from(triangles);
}

function distance(
  first: Pick<NormalizedLandmark, "x" | "y">,
  second: Pick<NormalizedLandmark, "x" | "y">,
  width: number,
  height: number,
) {
  return Math.hypot(
    (first.x - second.x) * width,
    (first.y - second.y) * height,
  );
}

function angleInDegrees(radians: number) {
  return radians === 0 ? 0 : (radians * 180) / Math.PI;
}

export function estimateHeadRotation(matrix?: Matrix): {
  pitch: number;
  yaw: number;
  roll: number;
} | null {
  if (
    !matrix ||
    matrix.rows !== 4 ||
    matrix.columns !== 4 ||
    matrix.data.length < 16
  ) {
    return null;
  }

  const values = matrix.data;
  const scaleX = Math.hypot(values[0], values[1], values[2]);
  const scaleY = Math.hypot(values[4], values[5], values[6]);
  const scaleZ = Math.hypot(values[8], values[9], values[10]);
  if (scaleX === 0 || scaleY === 0 || scaleZ === 0) return null;

  const yaw = Math.asin(
    Math.max(-1, Math.min(1, values[8] / scaleZ)),
  );
  const pitch = Math.atan2(-values[9] / scaleZ, values[10] / scaleZ);
  const roll = Math.atan2(-values[4] / scaleY, values[0] / scaleX);

  if (![pitch, yaw, roll].every(Number.isFinite)) {
    return null;
  }

  return {
    pitch: angleInDegrees(pitch),
    yaw: angleInDegrees(yaw),
    roll: angleInDegrees(roll),
  };
}

export function measureFace(
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
  matrix?: Matrix,
): FaceMeasurements | null {
  const requiredIndices = [10, 33, 133, 152, 234, 263, 362, 454];
  if (
    width <= 0 ||
    height <= 0 ||
    requiredIndices.some((index) => !landmarks[index])
  ) {
    return null;
  }

  const [top, leftEyeOuter, leftEyeInner, chin, cheekLeft, rightEyeOuter, rightEyeInner, cheekRight] =
    requiredIndices.map((index) => landmarks[index]);
  const leftEye = {
    x: (leftEyeOuter.x + leftEyeInner.x) / 2,
    y: (leftEyeOuter.y + leftEyeInner.y) / 2,
  };
  const rightEye = {
    x: (rightEyeOuter.x + rightEyeInner.x) / 2,
    y: (rightEyeOuter.y + rightEyeInner.y) / 2,
  };
  const rotation = estimateHeadRotation(matrix);

  return {
    faceWidth: distance(cheekLeft, cheekRight, width, height),
    faceHeight: distance(top, chin, width, height),
    eyeSpacing: distance(leftEye, rightEye, width, height),
    eyeLineAngle: angleInDegrees(
      Math.atan2(
        (rightEye.y - leftEye.y) * height,
        (rightEye.x - leftEye.x) * width,
      ),
    ),
    pitch: rotation?.pitch ?? null,
    yaw: rotation?.yaw ?? null,
    roll: rotation?.roll ?? null,
  };
}
