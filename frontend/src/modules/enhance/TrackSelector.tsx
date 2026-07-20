import type { AudioTrackInfo, SubtitleTrackInfo } from "../../lib/apiTypes";

interface TrackSelectorProps {
  audioTracks: AudioTrackInfo[];
  subtitleTracks: SubtitleTrackInfo[];
  selectedAudioIndices: number[];
  onChangeAudioIndices: (indices: number[]) => void;
  keepSubtitles: boolean;
  onChangeKeepSubtitles: (value: boolean) => void;
}

function channelLabel(channels: number): string {
  if (channels === 1) {
    return "mono";
  }
  if (channels === 2) {
    return "stereo";
  }
  return `${channels}ch`;
}

function trackLabel(track: AudioTrackInfo): string {
  const language = track.language ?? "unknown";
  return `${language} (${track.codec}, ${channelLabel(track.channels)})`;
}

// Toggling never re-sorts the selection: the caller (VideoPanel) relies on
// index 0 staying the primary track the backend enhances/restores, so a
// newly added track is appended at the end instead of merged in ascending
// numeric order.
function toggleTrackIndex(selected: number[], index: number): number[] {
  if (selected.includes(index)) {
    return selected.filter((candidate) => candidate !== index);
  }
  return [...selected, index];
}

function DefaultBadge() {
  return (
    <span className="rounded-sm bg-surface-2 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-text-faint">
      Default
    </span>
  );
}

function AudioTrackOption({
  track,
  isSelected,
  onToggle,
}: {
  track: AudioTrackInfo;
  isSelected: boolean;
  onToggle: (index: number) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-sm text-text">
      <input
        type="checkbox"
        checked={isSelected}
        onChange={() => onToggle(track.index)}
        aria-label={trackLabel(track)}
        className="h-3.5 w-3.5 accent-accent"
      />
      {trackLabel(track)}
      {track.isDefault && <DefaultBadge />}
    </label>
  );
}

export function TrackSelector({
  audioTracks,
  subtitleTracks,
  selectedAudioIndices,
  onChangeAudioIndices,
  keepSubtitles,
  onChangeKeepSubtitles,
}: TrackSelectorProps) {
  function handleToggleAudio(index: number) {
    onChangeAudioIndices(toggleTrackIndex(selectedAudioIndices, index));
  }

  return (
    <div className="flex flex-col gap-4">
      {audioTracks.length > 0 && (
        <fieldset className="flex flex-col gap-2">
          <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
            Audio tracks
          </legend>
          {audioTracks.map((track) => (
            <AudioTrackOption
              key={track.index}
              track={track}
              isSelected={selectedAudioIndices.includes(track.index)}
              onToggle={handleToggleAudio}
            />
          ))}
        </fieldset>
      )}
      {subtitleTracks.length > 0 && (
        <label className="flex items-center gap-2 text-sm text-text">
          <input
            type="checkbox"
            checked={keepSubtitles}
            onChange={(event) => onChangeKeepSubtitles(event.target.checked)}
            aria-label="Keep embedded subtitles"
            className="h-3.5 w-3.5 accent-accent"
          />
          Keep embedded subtitles ({subtitleTracks.length})
        </label>
      )}
    </div>
  );
}
