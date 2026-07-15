import { useQuery } from "@tanstack/react-query";
import { getEngineInfo } from "../../lib/api";
import type { VideoProfileResponse } from "../../lib/apiTypes";

interface VideoProfileControlsProps {
  value: string | null;
  onChange: (profile: VideoProfileResponse) => void;
}

interface ProfileGroup {
  label: string;
  profiles: VideoProfileResponse[];
}

function capitalize(word: string): string {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

function groupProfiles(profiles: VideoProfileResponse[]): ProfileGroup[] {
  const categories = [...new Set(profiles.map((profile) => profile.category))];
  return categories.map((category) => ({
    label: capitalize(category),
    profiles: profiles.filter((profile) => profile.category === category),
  }));
}

function formatProfileMeta(profile: VideoProfileResponse): string {
  return `${profile.scale}x · ${profile.videoCodec} · CRF ${profile.crf}`;
}

function profileOptionClassName(isSelected: boolean): string {
  const base =
    "flex cursor-pointer flex-col gap-1 rounded border px-3 py-2 transition-[background-color,border-color] duration-fast focus-within:outline focus-within:outline-2 focus-within:outline-accent";
  if (isSelected) {
    return `${base} border-accent bg-surface-2`;
  }
  return `${base} border-border bg-surface hover:border-text-faint`;
}

function ProfileOption({
  profile,
  isSelected,
  onChange,
}: {
  profile: VideoProfileResponse;
  isSelected: boolean;
  onChange: (profile: VideoProfileResponse) => void;
}) {
  return (
    <label className={profileOptionClassName(isSelected)}>
      <span className="flex items-center gap-2">
        <input
          type="radio"
          name="video-profile"
          value={profile.key}
          checked={isSelected}
          onChange={() => onChange(profile)}
          className="h-3.5 w-3.5 accent-accent"
        />
        <span className="text-sm text-text">{profile.label}</span>
      </span>
      <span className="pl-[22px] text-xs text-text-dim">{profile.description}</span>
      <span className="font-mono-tabular pl-[22px] text-xs text-text-dim">{formatProfileMeta(profile)}</span>
    </label>
  );
}

export function VideoProfileControls({ value, onChange }: VideoProfileControlsProps) {
  const engineQuery = useQuery({ queryKey: ["engine"], queryFn: getEngineInfo });

  if (engineQuery.isLoading) {
    return <p className="text-sm text-text-dim">Loading video profiles…</p>;
  }

  if (engineQuery.isError) {
    return <p className="text-sm text-danger">Could not load video profiles.</p>;
  }

  const groups = groupProfiles(engineQuery.data?.videoProfiles ?? []);

  return (
    <fieldset className="flex flex-col gap-4">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Profile</legend>
      {groups.map((group) => (
        <div key={group.label} role="group" aria-label={group.label} className="flex flex-col gap-2">
          <h3 className="text-xs font-medium text-text-faint">{group.label}</h3>
          {group.profiles.map((profile) => (
            <ProfileOption
              key={profile.key}
              profile={profile}
              isSelected={profile.key === value}
              onChange={onChange}
            />
          ))}
        </div>
      ))}
    </fieldset>
  );
}
