from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine
from app.services.media_tools import MediaTools


class VideoUpscaler:
    def __init__(self, settings: Settings, engine: RealEsrganNcnnEngine, media_tools: MediaTools) -> None:
        self.settings = settings
        self.engine = engine
        self.media_tools = media_tools

    def available(self) -> bool:
        return self.engine.available() and self.media_tools.available()

    async def run(self, job: VideoUpscaleJob) -> Path:
        if not self.available():
            raise RuntimeError("Video pipeline is not available. Ensure Real-ESRGAN and FFmpeg are installed.")

        work_dir = self.settings.video_work_path / job.id
        frames_in = work_dir / "frames-in"
        frames_out = work_dir / "frames-out"
        audio_path = work_dir / "audio.m4a"
        work_dir.mkdir(parents=True, exist_ok=True)
        frames_in.mkdir(parents=True, exist_ok=True)
        frames_out.mkdir(parents=True, exist_ok=True)

        probe = self.media_tools.ffprobe_json(job.source_path)
        video_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
        has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
        if not video_stream:
            raise RuntimeError("No video stream found in the uploaded file")

        fps = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "30/1"
        job.metadata["stage"] = "probing"
        job.metadata["fps"] = fps
        job.metadata["sourceWidth"] = int(video_stream.get("width") or 0)
        job.metadata["sourceHeight"] = int(video_stream.get("height") or 0)
        job.metadata["duration"] = float(probe.get("format", {}).get("duration") or 0)

        job.metadata["stage"] = "extracting_frames"
        await self._run_process(
            [
                self.settings.ffmpeg_binary,
                "-y",
                "-i",
                str(job.source_path),
                "-vsync",
                "0",
                "-threads",
                str(self.settings.ffmpeg_decode_threads),
                str(frames_in / "%08d.png"),
            ]
        )

        if job.keep_audio and has_audio:
            job.metadata["stage"] = "extracting_audio"
            await self._run_process(
                [
                    self.settings.ffmpeg_binary,
                    "-y",
                    "-i",
                    str(job.source_path),
                    "-vn",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(audio_path),
                ]
            )

        job.metadata["stage"] = "upscaling_frames"
        await self._run_process(
            [
                self.settings.engine_binary,
                "-i",
                str(frames_in),
                "-o",
                str(frames_out),
                "-n",
                job.model_name,
                "-s",
                str(job.scale),
                "-m",
                self.settings.engine_models_dir,
                "-f",
                "png",
                "-g",
                "0",
                "-j",
                f"2:{self.settings.cpu_fallback_workers}:{self.settings.cpu_fallback_workers}",
            ]
        )

        output_path = self.settings.outputs_path / f"{job.id}.{job.output_container}"
        encode_cmd = [
            self.settings.ffmpeg_binary,
            "-y",
            "-framerate",
            fps,
            "-i",
            str(frames_out / "%08d.png"),
        ]
        if job.keep_audio and audio_path.exists():
            encode_cmd += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0"]

        encode_cmd += self._build_video_encode_options(job)

        if job.keep_audio and audio_path.exists():
            encode_cmd += ["-c:a", "copy"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_cmd.append(str(output_path))

        job.metadata["stage"] = "encoding_video"
        await self._run_process(encode_cmd)

        if not output_path.exists():
            raise RuntimeError("Video processing finished but no output file was produced")

        job.metadata["stage"] = "completed"
        job.metadata["outputWidth"] = job.metadata["sourceWidth"] * job.scale
        job.metadata["outputHeight"] = job.metadata["sourceHeight"] * job.scale
        return output_path

    def _build_video_encode_options(self, job: VideoUpscaleJob) -> list[str]:
        options = [
            "-c:v",
            job.video_codec,
            "-preset",
            job.video_preset,
            "-crf",
            str(job.crf),
            "-pix_fmt",
            "yuv420p",
        ]

        if job.video_codec == "libx265":
            options += [
                "-x265-params",
                f"frame-threads=4:pools={self.settings.ffmpeg_x265_threads}",
                "-threads",
                str(min(self.settings.ffmpeg_x265_threads, 8)),
            ]
        else:
            options += ["-threads", str(self.settings.ffmpeg_encode_threads)]

        return options

    async def _run_process(self, command: list[str]) -> None:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(self._summarize_process_error(stderr, stdout))

    def _summarize_process_error(self, stderr: bytes, stdout: bytes) -> str:
        text = (stderr or stdout).decode("utf-8", errors="ignore")
        lowered = text.lower()

        if "cannot open libx265" in lowered or "incorrect parameters such as bit_rate, rate, width or height" in lowered:
            return (
                "FFmpeg failed while encoding with H.265/libx265. This usually happens because the selected "
                "x265 thread settings are invalid for the current input. Try H.264/libx264, or use a lower x265 "
                "thread count."
            )

        if "nothing was written into output file" in lowered:
            return "FFmpeg could not write the output video file. Check codec/container compatibility and selected options."

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else "External process failed"
