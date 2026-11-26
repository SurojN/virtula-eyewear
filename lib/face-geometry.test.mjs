import assert from "node:assert/strict";
import test from "node:test";
import {
  createFaceMeshIndices,
  estimateLandmarkConfidence,
  estimateHeadRotation,
  measureFace,
} from "./face-geometry.ts";

const identityMatrix = {
  rows: 4,
  columns: 4,
  data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
};

function makeLandmarks() {
  const landmarks = Array.from({ length: 478 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 1,
  }));
  landmarks[10] = { ...landmarks[10], x: 0.5, y: 0.1 };
  landmarks[152] = { ...landmarks[152], x: 0.5, y: 0.9 };
  landmarks[234] = { ...landmarks[234], x: 0.1, y: 0.5 };
  landmarks[454] = { ...landmarks[454], x: 0.9, y: 0.5 };
  landmarks[33] = { ...landmarks[33], x: 0.3, y: 0.4 };
  landmarks[133] = { ...landmarks[133], x: 0.4, y: 0.4 };
  landmarks[362] = { ...landmarks[362], x: 0.7, y: 0.42 };
  landmarks[263] = { ...landmarks[263], x: 0.6, y: 0.42 };
  return landmarks;
}

test("head rotation returns zero for an identity transform", () => {
  assert.deepEqual(estimateHeadRotation(identityMatrix), {
    pitch: 0,
    yaw: 0,
    roll: 0,
  });
});

test("head rotation extracts yaw from a 4x4 transform", () => {
  const radians = (30 * Math.PI) / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  const matrix = {
    rows: 4,
    columns: 4,
    data: [
      cosine * 1.8,
      0,
      -sine * 1.8,
      0,
      0,
      1.8,
      0,
      0,
      sine * 1.8,
      0,
      cosine * 1.8,
      0,
      0,
      0,
      0,
      1,
    ],
  };

  assert.ok(Math.abs(estimateHeadRotation(matrix).yaw - 30) < 1e-8);
});

test("face measurements use image-space pixel dimensions", () => {
  const result = measureFace(makeLandmarks(), 100, 100, identityMatrix);

  assert.ok(result);
  assert.equal(result.faceWidth, 80);
  assert.equal(result.faceHeight, 80);
  assert.ok(Math.abs(result.eyeSpacing - Math.hypot(30, 2)) < 1e-8);
  assert.ok(Math.abs(result.eyeLineAngle - (Math.atan2(2, 30) * 180) / Math.PI) < 1e-8);
});

test("face measurements reject incomplete landmarks and invalid dimensions", () => {
  assert.equal(measureFace([], 100, 100), null);
  assert.equal(measureFace(makeLandmarks(), 0, 100), null);
  assert.equal(estimateHeadRotation({ rows: 3, columns: 3, data: [] }), null);
});

test("face tessellation edges are assembled into unique triangles", () => {
  const indices = createFaceMeshIndices(
    [
      { start: 0, end: 1 },
      { start: 1, end: 2 },
      { start: 2, end: 0 },
      { start: 1, end: 0 },
      { start: 3, end: 4 },
    ],
    5,
  );

  assert.deepEqual([...indices], [0, 1, 2]);
});

test("landmark confidence uses presence only and defaults when unavailable", () => {
  assert.equal(estimateLandmarkConfidence(makeLandmarks(), [33, 263]), 1);
  const landmarks = makeLandmarks();
  landmarks[33].presence = 0.4;
  landmarks[263].presence = 0.8;
  assert.ok(Math.abs(estimateLandmarkConfidence(landmarks, [33, 263]) - 0.6) < 1e-8);
});
