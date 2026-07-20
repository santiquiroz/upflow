import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TrackSelector } from "./TrackSelector";

const audioTracks = [
  { index: 1, codec: "ac3", channels: 2, isDefault: true, language: "jpn" },
  { index: 2, codec: "aac", channels: 6, isDefault: false, language: "eng" },
];
const subtitleTracks = [{ index: 3, codec: "ass", language: "eng" }];

describe("TrackSelector", () => {
  it("shows one checkbox per audio track with language and default badge", () => {
    render(
      <TrackSelector
        audioTracks={audioTracks}
        subtitleTracks={subtitleTracks}
        selectedAudioIndices={[1]}
        onChangeAudioIndices={vi.fn()}
        keepSubtitles={false}
        onChangeKeepSubtitles={vi.fn()}
      />,
    );
    expect(screen.getByLabelText(/jpn/i)).toBeChecked();
    expect(screen.getByLabelText(/eng/i)).not.toBeChecked();
    expect(screen.getByText(/default/i)).toBeInTheDocument();
  });

  it("calls onChangeAudioIndices with the toggled track added", () => {
    const onChange = vi.fn();
    render(
      <TrackSelector
        audioTracks={audioTracks}
        subtitleTracks={subtitleTracks}
        selectedAudioIndices={[1]}
        onChangeAudioIndices={onChange}
        keepSubtitles={false}
        onChangeKeepSubtitles={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByLabelText(/eng/i));
    expect(onChange).toHaveBeenCalledWith([1, 2]);
  });

  it("calls onChangeAudioIndices with the toggled track removed", () => {
    const onChange = vi.fn();
    render(
      <TrackSelector
        audioTracks={audioTracks}
        subtitleTracks={subtitleTracks}
        selectedAudioIndices={[1, 2]}
        onChangeAudioIndices={onChange}
        keepSubtitles={false}
        onChangeKeepSubtitles={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByLabelText(/jpn/i));
    expect(onChange).toHaveBeenCalledWith([2]);
  });

  it("keeps the existing order instead of sorting so the primary track stays first", () => {
    const onChange = vi.fn();
    // Primary track deliberately has a HIGHER index (5) than the track being
    // added (2): an ascending sort would produce [2, 5], which is exactly
    // what this test must fail on if a `.sort()` is reintroduced.
    const reorderedTracks = [
      { index: 5, codec: "aac", channels: 2, isDefault: true, language: "jpn" },
      { index: 2, codec: "ac3", channels: 6, isDefault: false, language: "eng" },
    ];
    render(
      <TrackSelector
        audioTracks={reorderedTracks}
        subtitleTracks={[]}
        selectedAudioIndices={[5]}
        onChangeAudioIndices={onChange}
        keepSubtitles={false}
        onChangeKeepSubtitles={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByLabelText(/eng/i));
    expect(onChange).toHaveBeenCalledWith([5, 2]);
  });

  it("renders nothing for subtitles when there are no subtitle tracks", () => {
    render(
      <TrackSelector
        audioTracks={audioTracks}
        subtitleTracks={[]}
        selectedAudioIndices={[1]}
        onChangeAudioIndices={vi.fn()}
        keepSubtitles={false}
        onChangeKeepSubtitles={vi.fn()}
      />,
    );
    expect(screen.queryByLabelText(/subtitle/i)).not.toBeInTheDocument();
  });

  it("toggles keepSubtitles when the subtitle checkbox is present and clicked", () => {
    const onChange = vi.fn();
    render(
      <TrackSelector
        audioTracks={audioTracks}
        subtitleTracks={subtitleTracks}
        selectedAudioIndices={[1]}
        onChangeAudioIndices={vi.fn()}
        keepSubtitles={false}
        onChangeKeepSubtitles={onChange}
      />,
    );
    fireEvent.click(screen.getByLabelText(/subtitle/i));
    expect(onChange).toHaveBeenCalledWith(true);
  });
});
