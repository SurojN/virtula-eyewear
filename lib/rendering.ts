export type ViewportProjection = {
  x: number;
  y: number;
  screenX: number;
  screenY: number;
  coverScale: number;
};

export function projectFaceDepth(
  normalizedDepth: number,
  videoWidth: number,
  coverScale: number,
) {
  if (
    !Number.isFinite(normalizedDepth) ||
    !Number.isFinite(videoWidth) ||
    !Number.isFinite(coverScale) ||
    videoWidth <= 0 ||
    coverScale <= 0
  ) {
    return null;
  }
  return -normalizedDepth * videoWidth * coverScale;
}

export function projectMirroredPoint(
  x: number,
  y: number,
  videoWidth: number,
  videoHeight: number,
  viewportWidth: number,
  viewportHeight: number,
): ViewportProjection | null {
  if (
    ![x, y, videoWidth, videoHeight, viewportWidth, viewportHeight].every(
      Number.isFinite,
    ) ||
    videoWidth <= 0 ||
    videoHeight <= 0 ||
    viewportWidth <= 0 ||
    viewportHeight <= 0
  ) {
    return null;
  }

  const coverScale = Math.max(
    viewportWidth / videoWidth,
    viewportHeight / videoHeight,
  );
  const renderedWidth = videoWidth * coverScale;
  const renderedHeight = videoHeight * coverScale;
  const cropX = (viewportWidth - renderedWidth) / 2;
  const cropY = (viewportHeight - renderedHeight) / 2;
  const screenX = viewportWidth - (x * renderedWidth + cropX);
  const screenY = y * renderedHeight + cropY;

  return {
    x: screenX - viewportWidth / 2,
    y: viewportHeight / 2 - screenY,
    screenX,
    screenY,
    coverScale,
  };
}
