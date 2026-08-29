# Band Rehearsal Kit — plan de implementación

Rama: `feature/band-rehearsal-kit` (worktree aislado; la sesión 3D trabaja en el checkout principal — no tocar fuera de este worktree). Base: master v0.77.0.

Investigación completa: `~/Documents/Last30Days/ai-music-stem-separation-and-transcription-raw-v3.md` + memoria `project_upflow_band_rehearsal_kit`.

## Objetivo

Tres features para bandas ensayando con grabaciones reales:

- **F1 — Pistas minus-one**: separar multi-stem y generar "la canción sin X instrumento" por cada stem elegido, con guía opcional del instrumento removido a bajo volumen.
- **F2 — Karaoke por cantante**: etiquetar cada línea de la letra con su cantante (embeddings sobre el stem vocal + muestras de enrollment), mutear por líneas para practicar, subtítulos coloreados; modelo lead/backing para armonías.
- **F3 — Transcripción por stem**: cada stem → MIDI + MusicXML (+ tabs para guitarra/bajo), como borrador editable ("abrir en MuseScore/Guitar Pro").

## Decisiones cerradas (no re-litigar)

1. **Minus-one server-side**: pista horneada como archivo descargable (el caso de uso es llevarla al ensayo). Mezcla en vivo client-side = v2.
2. **Representación**: derivación a nivel pipeline (post-separación, pre-encode), NO nuevo `StemSource` en el catálogo. Ids derivados `minus_<stem_id>`.
3. **Matemática**: `minus = mix_decodificado − g·stem`, float64, `align_lengths`; `g = 1 − guide_percent/100`. Clip-guard (escalar solo si pico >1.0), NO normalización completa — preservar nivel original.
4. **Modelos F1**: arranca con `umx_4stem` ya instalado (infra pura, testeable CPU). BS-RoFormer-SW 6-stem = F1b, requiere export ONNX en repo port aparte + validación GPU coordinada con la sesión 3D.
5. **F2 modelo armonías**: `UVR_MDXNET_KARA_2.onnx` (arch mdx ya soportada, registro Case A directo). Mel-RoFormer Karaoke = upgrade posterior (necesita export).
6. **F2 embeddings**: reusar x-vector `tdnn.onnx` del pack voice-conversion vía protocolo `SpeakerEmbedder` abstracto (swap a wespeaker después si la calidad no da). Clustering con scipy (ya es dep) — sklearn NO.
7. **F2 orden**: ASR sobre mix (timings actuales) + embeddings sobre stem vocal. No reordenar transcripción.
8. **F3 superficie**: extiende AudioJob post-separación (opción 1 del mapa). No nueva familia de jobs.
9. **F3 serializers**: MIDI y MusicXML escritos a mano (sin mido/music21 — política zero-deps). Tabs: asignación cuerda/traste DP, guitarra 6 cuerdas estándar + bajo 4; afinaciones alternativas fuera de alcance v1.
10. **F3 modelo**: Basic Pitch ONNX (Apache-2.0) para stems con altura. Batería EXCLUIDA v1 (mejor ADT abierto es CC BY-NC) — nota explícita en UI. Transkun/beat_this = excepciones torch a evaluar en v2.
11. **Unísono real (dos cantantes misma melodía)**: imposible hoy — avisar en UI, ofrecer "mutear ambos".

## Contrato F1a (congelado — backend y frontend implementan contra esto)

- Form `POST /audio/jobs` (convención CSV de la familia audio): `practice_stems` = CSV de stem ids (`"drums,bass"`), `practice_guide_percent` = int 0–30, default 0.
- Validación en `audio_job_manager`: requiere `separate=true` + modelo con ≥3 stems; `practice_stems ⊆ spec.stem_ids()`; error estilo `missing_pack_message`.
- Derivados aparecen en `stems[]` del `AudioJobResponse` con id `minus_<id>` y label key `audio.stem.minus_<id>`; descargables vía `?stem=minus_<id>` (widen `_valid_stems_for` routes.py:2305 y `_audio_stem_downloads` :538 — sin esto 400, contradicción #1 del mapa).
- `AudioJobResponse`: `practiceStems: string[]`, `practiceGuidePercent: number`.
- Naming archivos: `{job.id}.minus_<stem>.{fmt}` (mismo loop de encode).
- Derivación corre después de `_run_ensemble` (audio_pipeline.py:332).
- MCP: `upflow_process_audio` + `practice_stems`, `practice_guide_percent` (enviar solo si truthy).
- Progreso: dentro de `finalizing` (no tocar la escalera de progress.py:151-154 en v1).

## Contrato F2a (congelado)

v1 SIN enrollment: clustering no supervisado sobre el stem vocal (el caso banda = cantantes que se turnan líneas). Enrollment con muestras por cantante = v2.

- Form karaoke create (convención repeated-fields de la familia karaoke): `detect_singers: bool` (default false) + `singer_count: int` 2–4 (default 2, solo válido con detect_singers).
- Prepare: retener el wav de vocals antes del descarte (karaoke_job_manager :422-426); por cada línea de letra ya alineada, embedding sobre esa ventana del stem vocal; clustering jerárquico coseno (scipy, ya dep) a `singer_count` clusters; etiqueta `s1..sN` por línea. Líneas demasiado cortas heredan el vecino más cercano en tiempo.
- Embeddings: protocolo `SpeakerEmbedder` abstracto. Primera opción: reusar x-vector `tdnn.onnx` del pack voice-conversion. Si el wrapper existente no sirve limpio para esto, pack nuevo `singer-embeddings` (Case B) con WeSpeaker CAM++ (URL/sha en 2026-08-28-model-artifacts.md, Apache-2.0, dim 512, fbank 80 @16kHz — resamplear la ventana).
- Schemas: `KaraokeLyricLine.singer: str|null`, `KaraokeLyricEdit.singer` (reasignación en review), `KaraokeJobResponse.singers: [{id, label}]` con rename vía el endpoint de edición existente.
- Render: campos opcionales `singer_colors` (por singer id) y `mute_singer: str|null`. Con `mute_singer`: el audio del render = instrumental + vocals con las líneas de ese cantante muteadas por tiempo (crossfades ~50 ms), y además se expone el audio-only como descarga. Subtítulos coloreados por cantante (estilos ASS por singer).
- UI Studio: badge de cantante por línea (reasignable), rename de cantantes, color pickers pre-render, selector "practicar como <cantante>" (mute), aviso de unísono (decisión #11).
- KARA_2 (lead/backing, armonías): registro Case A completo — spec mdx con stems `lead_vocals`/`backing_vocals` categoría karaoke, ps1 + sha de model-artifacts.md, créditos, i18n, tests anti-drift. Disponible como modelo más en el picker; la integración "mutear sub-stem en sección armonizada" queda v2.

## Fases

| Fase | Contenido | Estado |
|---|---|---|
| F1a | Infra minus-one completa (backend+frontend+tests) con umxhq | en curso |
| F2a | KARA_2 registro + per-singer (embeddings, clustering, ASS colores, UI studio) | pendiente |
| F3a | Pack basic-pitch + engine AMT + writers MIDI/MusicXML/tab + UI | pendiente |
| F1b | Export ONNX BS-RoFormer-SW (repo port aparte) + registro + DrumSep | pendiente, GPU coordinada |

Anclas de código exactas por fase: ver mapa del workflow (journal `wf_42c6fd4e-27f`) — resumen clave: pipeline :311-322, routes :538/:1137/:2305, job manager :59-145, karaoke manager :363-384/:422-426 (el stem vocal se borra ahí — retenerlo para embeddings), specs Case A/B en el mapa.

## Reglas de ejecución

- TDD: tests primero, convenciones existentes (fakes `_create_session`, `FakeSeparator` de `tests.test_audio_karaoke`, wavs sintéticos `subtype="FLOAT"` con RNG seedeado).
- i18n SIEMPRE en ambos locales + `i18n.test.ts` de paridad.
- Cero deps pip nuevas sin "Excepcion aprobada".
- Commits por hito con formato Historia técnica del repo.
- GPU: nada pesado sin coordinar (sesión 3D comparte la RX 7800 XT).
