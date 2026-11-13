import { CanvasTexture, SRGBColorSpace, Sprite, SpriteMaterial } from 'three';
import { TITLE_FONT, TITLE_LINE } from '../constants';
import { noRaycast } from './raycast';

// Title text is drawn on canvases and shown as sprites: flat images that always face the camera.
const TITLE_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif';
const TITLE_PAD = 14;

const measureCtx = document.createElement('canvas').getContext('2d')!;
measureCtx.font = `600 ${TITLE_FONT}px ${TITLE_FAMILY}`;

export interface TitleDims {
  w: number;
  h: number;
}

export function titleDims(lines: string[]): TitleDims {
  return {
    w: Math.ceil(Math.max(...lines.map((l) => measureCtx.measureText(l).width))) + TITLE_PAD * 2,
    h: lines.length * TITLE_LINE + TITLE_PAD * 2,
  };
}

export function makeTitleSprite(lines: string[], dims: TitleDims): Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = dims.w;
  canvas.height = dims.h;
  const ctx = canvas.getContext('2d')!;
  ctx.font = `600 ${TITLE_FONT}px ${TITLE_FAMILY}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  ctx.lineWidth = TITLE_FONT * 0.2;
  ctx.strokeStyle = 'rgba(8, 10, 16, 0.92)'; // a dark outline keeps the text readable on any color
  ctx.fillStyle = '#ffffff';
  lines.forEach((line, i) => {
    const y = TITLE_PAD + TITLE_LINE * (i + 0.5) + 2;
    ctx.strokeText(line, dims.w / 2, y);
    ctx.fillText(line, dims.w / 2, y);
  });
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  // Always drawn on top, even where an open bubble's papers are.
  const sprite = new Sprite(
    new SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false }),
  );
  sprite.renderOrder = 20;
  sprite.raycast = noRaycast; // text must not block hovering or clicking what's underneath
  return sprite;
}
