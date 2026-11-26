import assert from "node:assert/strict";
import test from "node:test";
import { projectFaceDepth, projectMirroredPoint } from "./rendering.ts";

test("face depth uses the same cover-scaled camera coordinates as landmarks", () => {
  assert.equal(projectFaceDepth(-0.02, 1280, 0.5), 12.8);
  assert.equal(projectFaceDepth(-0.02, 1280, 1), 25.6);
  assert.equal(projectFaceDepth(-0.02, 0, 1), null);
});

test("projection mirrors a centered point in an aspect-matched viewport", () => {
  assert.deepEqual(
    projectMirroredPoint(0.5, 0.5, 1280, 720, 1280, 720),
    { x: 0, y: 0, screenX: 640, screenY: 360, coverScale: 1 },
  );
});

test("projection preserves cover cropping and mirrored horizontal position", () => {
  const projected = projectMirroredPoint(0.25, 0.5, 1280, 720, 720, 720);

  assert.ok(projected);
  assert.equal(projected.coverScale, 1);
  assert.equal(projected.screenX, 680);
  assert.equal(projected.x, 320);
  assert.equal(projected.y, 0);
});

test("projection rejects invalid source or viewport dimensions", () => {
  assert.equal(projectMirroredPoint(0.5, 0.5, 0, 720, 100, 100), null);
});
