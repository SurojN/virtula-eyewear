# Virtual Eyewear Studio

The fitting room includes a bundled GLB frame, MediaPipe face landmarks and
pose tracking, Three.js rendering, manual frame calibration, and upload support
for GLB models and PNG/JPG/WebP images.

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and allow camera access.
Deployed sites must use HTTPS. The MediaPipe model and WebAssembly runtime are
loaded from their configured remote URLs.

## Asset behavior

- **GLB** is the 3D option. Models are loaded with embedded textures, normalized
  by their measured width, and attached to tracked eye position, head
  orientation, and estimated bridge depth. Face and model depth share the
  preview's cover-scaled coordinates, with a small front-frame clearance to
  avoid false occlusion. Use **Adjust** to calibrate a model whose origin,
  scale, or orientation differs.
- **PNG/JPG/WebP** is a textured flat image. It is aligned to the face and fades
  as the head turns; it does not gain hidden frame depth, temples, or genuine
  side geometry. Use GLB for multi-angle fitting.
- A tessellated face depth surface provides approximate facial occlusion for 3D
  assets. It is based on MediaPipe landmarks, not a personalized scanned head
  mesh, so it is not a guarantee of physically exact occlusion at profile
  angles.
- Uploaded images are limited to 15 MB and 8192 × 8192 pixels; GLB uploads are
  limited to 50 MB. Uploads and camera frames remain on-device.
- The included starter GLB is used for catalog selections until each product
  has its own model asset.

## Geometry tests

With Node.js 22.6 or newer:

```bash
node --experimental-strip-types --test lib/*.test.mjs
```

## Manual webcam validation

Test with a real webcam; unit tests do not establish visual tracking quality.

- [ ] Check the bundled GLB loads and follows the face in a frontal view.
- [ ] Upload a transparent PNG and use scale/position/roll controls to align it.
- [ ] Confirm the image fades as you turn and does not claim to show side geometry.
- [ ] Upload a valid GLB, calibrate it, then test 30°, 45°, and 60° left/right.
- [ ] Check the near-side frame, far-side temple, bridge, and face occlusion
      while turning; record any model-specific drift or clipping.
- [ ] Look up/down, move closer/farther, and test temporary face occlusion.
- [ ] Stop/restart the camera, deny permission, and disconnect the camera.
- [ ] Test low light and verify recovery after the face returns to view.

Pose-based rendering and the landmark depth mesh are approximations. The
current build has not been visually validated across devices or with arbitrary
third-party GLB models.
