import { describe, expect, it } from "vitest";
import type { CctvBox } from "../../../services/cctv";
import {
  activeBoxes,
  boxAt,
  EMPTY_REDACTION,
  MAX_REDACTION_TRACKS,
  rangesText,
  redactionBlockerKey,
  redactionRequest,
  redactionSpan,
  uncoveredRanges,
  withEditedBoxes,
  withKeyframe,
  withKeyframeRemoved,
  withSpanFollowed,
  withStyle,
  withTrackAdded,
  withTrackEnd,
  withTrackRemoved,
  withTrackStart,
  type RedactionChoice,
  type RedactionTrack,
} from "./cctvRedaction";

const MOVING: RedactionTrack = {
  id: 1,
  firstFrame: 10,
  lastFrame: 30,
  keyframes: [
    { frame: 12, box: [0, 0, 20, 20] },
    { frame: 22, box: [100, 50, 40, 30] },
  ],
};

function withOneBox(box: CctvBox = [10, 10, 40, 40], frame = 5): RedactionChoice {
  return withTrackAdded(EMPTY_REDACTION, box, frame, [0, 99]);
}

describe("boxAt", () => {
  it("holds the box before the first keyframe and after the last, inside the box's frames", () => {
    expect(boxAt(MOVING, 10)).toEqual([0, 0, 20, 20]);
    expect(boxAt(MOVING, 30)).toEqual([100, 50, 40, 30]);
  });

  it("hides the box outside its frames", () => {
    expect(boxAt(MOVING, 9)).toBeNull();
    expect(boxAt(MOVING, 31)).toBeNull();
  });

  it("moves in a straight line between keyframes and rounds halves up like the backend", () => {
    expect(boxAt(MOVING, 17)).toEqual([50, 25, 30, 25]);
    const halfway: RedactionTrack = {
      id: 2,
      firstFrame: 0,
      lastFrame: 2,
      keyframes: [
        { frame: 0, box: [0, 0, 2, 2] },
        { frame: 2, box: [1, 1, 3, 3] },
      ],
    };
    expect(boxAt(halfway, 1)).toEqual([1, 1, 3, 3]);
  });
});

describe("editing boxes on a frame", () => {
  it("draws a new box that covers the whole span with one keyframe", () => {
    const choice = withOneBox([10, 10, 40, 40], 5);

    expect(choice.tracks).toEqual([{ id: 1, firstFrame: 0, lastFrame: 99, keyframes: [{ frame: 5, box: [10, 10, 40, 40] }] }]);
    expect(activeBoxes(choice, 80)).toEqual([{ id: 1, number: 1, box: [10, 10, 40, 40], isKeyframe: false }]);
  });

  it("turns a moved box into a keyframe on that frame, kept in frame order", () => {
    const choice = withOneBox();
    const active = activeBoxes(choice, 50);

    const moved = withEditedBoxes(choice, active, [[60, 10, 40, 40]], 50, [0, 99]);
    const earlier = withKeyframe(moved, 1, 2, [0, 0, 40, 40]);

    expect(earlier.tracks[0].keyframes.map((keyframe) => keyframe.frame)).toEqual([2, 5, 50]);
    expect(boxAt(moved.tracks[0], 50)).toEqual([60, 10, 40, 40]);
  });

  it("removes the box the editor deleted and leaves the others", () => {
    const two = withTrackAdded(withOneBox(), [200, 10, 30, 30], 5, [0, 99]);
    const active = activeBoxes(two, 5);

    const left = withEditedBoxes(two, active, [[200, 10, 30, 30]], 5, [0, 99]);

    expect(left.tracks.map((track) => track.id)).toEqual([2]);
  });

  it("ignores an editor change that did not move anything", () => {
    const choice = withOneBox();

    expect(withEditedBoxes(choice, activeBoxes(choice, 5), [[10, 10, 40, 40]], 5, [0, 99])).toBe(choice);
  });

  it("stops at the box limit", () => {
    let choice = EMPTY_REDACTION;
    for (let index = 0; index < MAX_REDACTION_TRACKS + 2; index += 1) {
      choice = withTrackAdded(choice, [index, 0, 4, 4], 0, [0, 9]);
    }

    expect(choice.tracks).toHaveLength(MAX_REDACTION_TRACKS);
  });

  it("never removes the last keyframe", () => {
    const choice = withOneBox();

    expect(withKeyframeRemoved(choice, 1, 5)).toEqual(choice);
    const two = withKeyframe(choice, 1, 9, [0, 0, 8, 8]);
    expect(withKeyframeRemoved(two, 1, 9).tracks[0].keyframes).toEqual([{ frame: 5, box: [10, 10, 40, 40] }]);
  });

  it("deletes a whole box", () => {
    expect(withTrackRemoved(withOneBox(), 1).tracks).toEqual([]);
  });
});

describe("start and end of a box", () => {
  const choice: RedactionChoice = { style: "pixelate", tracks: [MOVING], nextId: 2 };

  it("drops keyframes that fall outside the new frames", () => {
    const started = withTrackStart(choice, 1, 15);

    expect(started.tracks[0]).toMatchObject({ firstFrame: 15, lastFrame: 30 });
    expect(started.tracks[0].keyframes.map((keyframe) => keyframe.frame)).toEqual([22]);
  });

  it("pins the box that was on screen when no keyframe is left inside", () => {
    const ended = withTrackEnd(choice, 1, 11);

    expect(ended.tracks[0]).toMatchObject({ firstFrame: 10, lastFrame: 11 });
    expect(ended.tracks[0].keyframes).toEqual([{ frame: 11, box: [0, 0, 20, 20] }]);
  });

  it("stretches the other end when a box starts after it ended", () => {
    const started = withTrackStart(choice, 1, 40);

    expect(started.tracks[0]).toMatchObject({ firstFrame: 40, lastFrame: 40 });
    expect(started.tracks[0].keyframes).toEqual([{ frame: 40, box: [100, 50, 40, 30] }]);
  });
});

describe("boxes that follow the trim", () => {
  const drawnInTrim = withTrackAdded(EMPTY_REDACTION, [10, 10, 40, 40], 150, [100, 200]);

  it("a box drawn over the whole trim grows with it, so the new frames stay hidden", () => {
    const widened = withSpanFollowed(drawnInTrim, [100, 200], [0, 300]);

    expect(widened.tracks[0]).toMatchObject({ firstFrame: 0, lastFrame: 300 });
    expect(widened.tracks[0].keyframes).toEqual([{ frame: 150, box: [10, 10, 40, 40] }]);
    expect(uncoveredRanges(widened, [0, 300])).toEqual([]);
  });

  it("a box drawn over the whole trim moves with a trim that moves", () => {
    const moved = withSpanFollowed(drawnInTrim, [100, 200], [400, 500]);

    expect(moved.tracks[0]).toMatchObject({ firstFrame: 400, lastFrame: 500 });
    expect(moved.tracks[0].keyframes).toEqual([{ frame: 400, box: [10, 10, 40, 40] }]);
  });

  it("a box with its own frames keeps them when the trim grows and is cut when it shrinks", () => {
    const own = withTrackAdded(EMPTY_REDACTION, [10, 10, 40, 40], 150, [120, 180]);

    expect(withSpanFollowed(own, [100, 200], [0, 300]).tracks[0]).toMatchObject({ firstFrame: 120, lastFrame: 180 });
    expect(withSpanFollowed(own, [100, 200], [130, 170]).tracks[0]).toMatchObject({ firstFrame: 130, lastFrame: 170 });
  });

  it("a box left entirely outside the trim is kept as it was and blocks Start", () => {
    const own = withTrackAdded(EMPTY_REDACTION, [10, 10, 40, 40], 150, [120, 180]);
    const outside = withSpanFollowed(own, [100, 200], [300, 400]);

    expect(outside.tracks[0]).toMatchObject({ firstFrame: 120, lastFrame: 180 });
    expect(redactionBlockerKey(outside, [300, 400])).toBe("cctv.redact.blocked.outsideTrim");
  });

  it("lists the frames of the copy that no box covers", () => {
    const own = withTrackAdded(EMPTY_REDACTION, [10, 10, 40, 40], 150, [120, 180]);
    const gaps = uncoveredRanges(withTrackAdded(own, [0, 0, 8, 8], 190, [190, 190]), [100, 200]);

    expect(gaps).toEqual([
      [100, 119],
      [181, 189],
      [191, 200],
    ]);
    expect(rangesText([[5, 5], [7, 9]])).toBe("5, 7–9");
  });

  it("uses the whole video when there is no trim", () => {
    expect(redactionSpan(null, 750)).toEqual([0, 749]);
    expect(redactionSpan([3, 9], 750)).toEqual([3, 9]);
  });
});

describe("request", () => {
  it("blocks Start until there is a box and while a box misses the trim", () => {
    expect(redactionBlockerKey(EMPTY_REDACTION, [0, 99])).toBe("cctv.redact.blocked.noBoxes");
    expect(redactionBlockerKey(withOneBox(), [0, 99])).toBeNull();
    expect(redactionBlockerKey(withOneBox(), [99, 200])).toBeNull();
    expect(redactionBlockerKey(withOneBox(), [100, 200])).toBe("cctv.redact.blocked.outsideTrim");
  });

  it("starts with the solid box, the only style that leaves nothing to recover", () => {
    expect(EMPTY_REDACTION.style).toBe("fill");
  });

  it("sends the style and each box with its keyframes", () => {
    const request = redactionRequest(withStyle(withOneBox(), "blur"));

    expect(request).toEqual({
      style: "blur",
      tracks: [{ firstFrame: 0, lastFrame: 99, keyframes: [{ frame: 5, box: [10, 10, 40, 40] }] }],
    });
  });
});
