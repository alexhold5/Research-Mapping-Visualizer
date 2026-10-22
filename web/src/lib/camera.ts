import type { PerspectiveCamera, Vector3 } from 'three';

/**
 * The slice of three's OrbitControls we touch. `graph.controls()` is typed as a bare `object`
 * by the library, so we assert to this instead of pulling in the full OrbitControls type.
 */
export interface OrbitLikeControls {
  target: Vector3;
  enableRotate: boolean;
  enablePan: boolean;
  screenSpacePanning: boolean;
  minDistance: number;
  maxDistance: number;
  mouseButtons: { LEFT: number; MIDDLE: number; RIGHT: number };
  touches: { ONE: number; TWO: number };
  addEventListener(type: 'change', listener: () => void): void;
  removeEventListener(type: 'change', listener: () => void): void;
}

/** three's MOUSE and TOUCH enums, inlined so we don't import the whole library for two numbers. */
export const MOUSE = { ROTATE: 0, DOLLY: 1, PAN: 2 } as const;
export const TOUCH = { ROTATE: 0, PAN: 1, DOLLY_PAN: 2, DOLLY_ROTATE: 3 } as const;

export interface Coords {
  x: number;
  y: number;
  z: number;
}

/** Where the camera sits and what it looks at — enough to describe any view of the map. */
export interface CameraPose {
  position: Coords;
  target: Coords;
}

export interface FlightController {
  flyTo(to: CameraPose, ms: number): void;
  cancel(): void;
  isFlying(): boolean;
}

const sub = (a: Coords, b: Coords): Coords => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const length = (v: Coords) => Math.hypot(v.x, v.y, v.z);

function normalize(v: Coords): Coords {
  const l = length(v) || 1;
  return { x: v.x / l, y: v.y / l, z: v.z / l };
}

/**
 * Smooth camera move that stays pointed at the map the whole way.
 *
 * The library's own cameraPosition() animation moves the camera and its look-at point on
 * different timers, which tilts the view mid-flight. This interpolates the *offset* from the
 * target instead: direction and distance separately, with distance in log space so the zoom
 * feels even. A flight that doesn't change the viewing angle (every flight in flat mode) leaves
 * the direction untouched and is a pure pan-and-zoom. Any user pan/zoom cancels the flight.
 */
export function createFlightController(
  camera: PerspectiveCamera,
  controls: OrbitLikeControls,
): FlightController {
  let flight: object | null = null;

  return {
    flyTo(to, ms) {
      const fromTarget: Coords = { x: controls.target.x, y: controls.target.y, z: controls.target.z };
      const fromOffset = sub(camera.position, fromTarget);
      const fromLen = length(fromOffset) || 1;
      const fromDir = normalize(fromOffset);

      const toOffset = sub(to.position, to.target);
      const toLen = length(toOffset) || 1;
      const toDir = normalize(toOffset);

      const start = performance.now();
      const mine = (flight = {});

      const step = (now: number) => {
        if (flight !== mine) return; // cancelled or superseded
        const p = Math.min(1, (now - start) / ms);
        const e = 1 - Math.pow(1 - p, 3); // ease-out cubic

        const tx = fromTarget.x + (to.target.x - fromTarget.x) * e;
        const ty = fromTarget.y + (to.target.y - fromTarget.y) * e;
        const tz = fromTarget.z + (to.target.z - fromTarget.z) * e;

        const dir = normalize({
          x: fromDir.x + (toDir.x - fromDir.x) * e,
          y: fromDir.y + (toDir.y - fromDir.y) * e,
          z: fromDir.z + (toDir.z - fromDir.z) * e,
        });
        const len = fromLen * Math.pow(toLen / fromLen, e);

        camera.position.set(tx + dir.x * len, ty + dir.y * len, tz + dir.z * len);
        controls.target.set(tx, ty, tz);

        if (p < 1) requestAnimationFrame(step);
        else flight = null;
      };
      requestAnimationFrame(step);
    },
    cancel() {
      flight = null;
    },
    isFlying() {
      return flight !== null;
    },
  };
}
