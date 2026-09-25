# stderr grabado de ffmpeg

Salida de error del ffmpeg vendorizado (`N-123588-g9c63742425-20260323`) para los
parsers de `app/services/video_analysis.py`. Cada archivo sale de un clip sintético
(`CLIPS` en `tests/test_video_analysis.py`) pasado por el mismo comando que arma el
módulo (`FIXTURE_RECIPES`). Se graba con ruta relativa y `cwd` temporal, así que no
guarda rutas de la máquina.

`test_recorded_fixture_still_matches_what_the_binary_prints` vuelve a correr cada
receta con el binario actual y compara lo que parsea contra el archivo: si una build
nueva cambia el formato o los valores, ese test falla.

Para regrabar (desde la raíz del repo):

```powershell
$env:FFMPEG_BINARY = "<ruta a ffmpeg.exe>"
.\.venv\Scripts\python -c "import sys; sys.path.insert(0, 'tests'); from pathlib import Path; import tempfile, test_video_analysis as t; t.record_all_fixtures(Path(tempfile.mkdtemp()))"
```
