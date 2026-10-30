export function maskHasPixels(rgba: Uint8ClampedArray): boolean {
  for (let offset = 3; offset < rgba.length; offset += 4) {
    if (rgba[offset] > 0) {
      return true;
    }
  }
  return false;
}

export function maskToOpaquePixels(rgba: Uint8ClampedArray): Uint8ClampedArray {
  const pixels = new Uint8ClampedArray(rgba.length);
  for (let offset = 0; offset < rgba.length; offset += 4) {
    const value = rgba[offset + 3] > 0 ? 255 : 0;
    pixels[offset] = value;
    pixels[offset + 1] = value;
    pixels[offset + 2] = value;
    pixels[offset + 3] = 255;
  }
  return pixels;
}

export function subtractProtectedPixels(
  editRgba: Uint8ClampedArray,
  protectRgba: Uint8ClampedArray,
): Uint8ClampedArray {
  if (editRgba.length !== protectRgba.length) {
    throw new RangeError("Mask pixel lengths must match");
  }
  const pixels = editRgba.slice();
  for (let offset = 3; offset < pixels.length; offset += 4) {
    if (protectRgba[offset] > 0) {
      pixels[offset] = 0;
    }
  }
  return pixels;
}
