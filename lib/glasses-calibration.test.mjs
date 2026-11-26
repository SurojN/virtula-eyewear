import assert from "node:assert/strict";
import test from "node:test";
import {
  calibrateModelBounds,
  estimateFrameScale,
} from "./glasses-calibration.ts";

test("frame scale compensates once for eye foreshortening at yaw", () => {
  const frontalScale = estimateFrameScale(200, 0);
  const sideScale = estimateFrameScale(100, 60);

  assert.equal(frontalScale, 100);
  assert.ok(Math.abs(sideScale - frontalScale) < 1e-8);
});

test("frame scale rejects invalid camera measurements", () => {
  assert.equal(estimateFrameScale(0, 0), null);
  assert.equal(estimateFrameScale(200, Number.NaN), null);
});

test("calibration fits model width and anchors its front plane at zero depth", () => {
  const result = calibrateModelBounds({
    min: { x: -1, y: -0.5, z: -2 },
    max: { x: 1, y: 0.5, z: 0.25 },
  });

  assert.ok(result);
  assert.equal(result.scale, 1.775);
  assert.equal(result.position.x, 0);
  assert.equal(result.position.y, 0);
  assert.equal(result.position.z, -0.44375);
  assert.equal((0.25 * result.scale) + result.position.z, 0);
});

test("calibration rejects empty or invalid model bounds", () => {
  assert.equal(
    calibrateModelBounds({
      min: { x: 0, y: 0, z: 0 },
      max: { x: 0, y: 1, z: 1 },
    }),
    null,
  );
  assert.equal(
    calibrateModelBounds({
      min: { x: 0, y: 0, z: 0 },
      max: { x: 1, y: 1, z: 1 },
    }, 0),
    null,
  );
});
