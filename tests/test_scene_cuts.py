from __future__ import annotations

from pathlib import Path

from app.services.scene_cuts import (
    build_scene_detect_command,
    frames_straddling_cut,
    parse_scene_cut_times,
    repair_interpolated_cuts,
    source_index_at_time,
)

# ---------------------------------------------------------------------------
# Medido (2026-08-05, `scripts/spike_scenecut.py`): con dos escenas pegadas —dos
# cuadros rojos y dos azules— RIFE produce un cuadro que queda a 144 del rojo y
# a 26 del azul, cuando los limpios quedan a menos de 1,2 de su color. O sea que
# inventa una mezcla de dos imagenes que no tienen nada que ver.
#
# El arreglo no es interpolar mejor: en un corte NO HAY movimiento que estimar.
# Es duplicar el ultimo cuadro de la escena que termina.
# ---------------------------------------------------------------------------


class TestParseSceneCutTimes:
    def test_reads_the_times_ffmpeg_printed(self) -> None:
        salida = (
            "[Parsed_scdet_0 @ 0000021] lavfi.scd.score: 15.625, lavfi.scd.time: 1\n"
            "[Parsed_scdet_0 @ 0000021] lavfi.scd.score: 42.0, lavfi.scd.time: 3.5\n"
        )

        assert parse_scene_cut_times(salida) == [1.0, 3.5]

    def test_a_video_without_cuts_gives_no_times(self) -> None:
        assert parse_scene_cut_times("frame= 100 fps=0.0 q=-1.0 size=N/A\n") == []

    def test_ignores_a_score_line_without_a_time(self) -> None:
        assert parse_scene_cut_times("lavfi.scd.score: 15.625\n") == []


class TestSourceIndexAtTime:
    def test_a_time_maps_to_the_frame_that_starts_the_new_scene(self) -> None:
        # A 24 fps, el segundo 1 es el cuadro 24.
        assert source_index_at_time(1.0, fps=24.0) == 24

    def test_rounds_instead_of_truncating(self) -> None:
        # 0,9999 s a 24 fps es el cuadro 24, no el 23: truncar correria el corte
        # un cuadro y el fantasma quedaria igual.
        assert source_index_at_time(0.9999, fps=24.0) == 24

    def test_time_zero_is_the_first_frame(self) -> None:
        assert source_index_at_time(0.0, fps=24.0) == 0


class TestFramesStraddlingCut:
    def test_finds_the_frame_that_mixes_the_two_scenes(self) -> None:
        # Lo medido: 4 cuadros de entrada, 7 de salida, corte antes del cuadro 2.
        # Las posiciones de salida son 0; 0,5; 1; 1,5; 2; 2,5; 3 y la unica que
        # cae ENTRE 1 y 2 es la cuarta.
        assert frames_straddling_cut(cut_index=2, source_count=4, output_count=7) == [3]

    def test_a_cut_on_an_exact_output_frame_needs_no_repair(self) -> None:
        # Con multiplicador entero y corte alineado, ningun cuadro es mezcla.
        assert frames_straddling_cut(cut_index=2, source_count=4, output_count=4) == []

    def test_a_bigger_multiplier_leaves_more_mixed_frames(self) -> None:
        # x4: entre dos cuadros de origen hay tres inventados.
        assert len(frames_straddling_cut(cut_index=1, source_count=3, output_count=9)) == 3

    def test_a_cut_at_the_first_frame_has_nothing_before_it(self) -> None:
        assert frames_straddling_cut(cut_index=0, source_count=4, output_count=7) == []

    def test_a_cut_past_the_end_is_ignored(self) -> None:
        assert frames_straddling_cut(cut_index=99, source_count=4, output_count=7) == []


class TestBuildSceneDetectCommand:
    def test_decodes_without_encoding_anything(self) -> None:
        comando = build_scene_detect_command(
            ffmpeg="ffmpeg.exe", video=Path("v.mp4"), threshold=10.0
        )

        assert comando[-1] == "-"
        assert "null" in comando

    def test_carries_the_threshold_into_the_filter(self) -> None:
        comando = build_scene_detect_command(
            ffmpeg="ffmpeg.exe", video=Path("v.mp4"), threshold=12.5
        )

        assert any("scdet" in arg and "12.5" in arg for arg in comando)


class TestRepairInterpolatedCuts:
    def make_frames(self, carpeta: Path, contenidos: list[bytes]) -> None:
        carpeta.mkdir(parents=True, exist_ok=True)
        for i, contenido in enumerate(contenidos, start=1):
            (carpeta / f"{i:08d}.png").write_bytes(contenido)

    def read_frames(self, carpeta: Path) -> list[bytes]:
        return [p.read_bytes() for p in sorted(carpeta.glob("*.png"))]

    def test_the_mixed_frame_is_replaced_by_the_one_before_it(self, tmp_path: Path) -> None:
        carpeta = tmp_path / "out"
        self.make_frames(carpeta, [b"A", b"A", b"A", b"MEZCLA", b"B", b"B", b"B"])

        reparados = repair_interpolated_cuts(
            carpeta, cut_indices=[2], source_count=4, output_count=7
        )

        assert reparados == 1
        assert self.read_frames(carpeta) == [b"A", b"A", b"A", b"A", b"B", b"B", b"B"]

    def test_a_video_without_cuts_is_left_untouched(self, tmp_path: Path) -> None:
        carpeta = tmp_path / "out"
        original = [b"1", b"2", b"3", b"4"]
        self.make_frames(carpeta, original)

        reparados = repair_interpolated_cuts(
            carpeta, cut_indices=[], source_count=2, output_count=4
        )

        assert reparados == 0
        assert self.read_frames(carpeta) == original

    def test_a_missing_frame_does_not_kill_the_job(self, tmp_path: Path) -> None:
        # Un indice que no existe en disco no puede tirar abajo un video entero
        # por un cuadro.
        carpeta = tmp_path / "out"
        self.make_frames(carpeta, [b"A", b"B"])

        assert repair_interpolated_cuts(
            carpeta, cut_indices=[2], source_count=4, output_count=7
        ) == 0
