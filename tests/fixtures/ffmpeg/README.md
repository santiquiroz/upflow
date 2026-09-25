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

## Capacidades de la build (`ffmpeg_capabilities`)

`version_gpl.txt`, `buildconf_gpl.txt`, `filters_gpl.txt` y `encoders_gpl.txt` son la
salida estándar de `-version`, `-hide_banner -buildconf`, `-hide_banner -filters` y
`-hide_banner -encoders` del mismo binario vendorizado. Los `*_lgpl.txt` son esas mismas
salidas editadas para simular una build sin GPL: se quitan los flags GPL del configure
(`--enable-gpl`, `libx264`, `libx265`, `libxvid`, `libxavs2`, `libdavs2`, `libvidstab`,
`frei0r`, `librubberband`), los filtros que dependen de GPL (`hqdn3d`, `fspp`, `spp`, `pp7`,
`uspp`, `eq`, entre otros) y los encoders `libx264`, `libx264rgb`, `libx265`, `libxvid` y
`libxavs2`. Las listas exactas están en `GPL_ONLY_FILTERS`, `GPL_ONLY_ENCODERS` y
`GPL_CONFIGURE_FLAGS` de `tests/test_ffmpeg_capabilities.py`.

`test_the_real_binary_matches_the_recorded_gpl_fixture` compara el binario actual con los
`*_gpl.txt`; si cambia la versión se saltea con el motivo. Para regrabar todo:

```powershell
.\.venv\Scripts\python -c "import sys; sys.path.insert(0, 'tests'); from app.config import Settings; import test_ffmpeg_capabilities as t; t.record_fixtures(Settings().ffmpeg_binary_path)"
```
