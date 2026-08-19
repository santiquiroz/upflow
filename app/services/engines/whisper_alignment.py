"""Tiempos por PALABRA a partir de las cross-attentions del decoder.

Whisper sabe en que segundo cae cada palabra, pero esa informacion no sale de los
tokens: sale de alinear los pesos de atencion cruzada con DTW. El grafo ONNX
publicado no expone esos pesos como salidas —aunque SI los calcula—, asi que el
primer paso es promover esos tensores a salidas del grafo.

El algoritmo y sus porques estan documentados en el port propio, que es donde se
valido contra transformers (1.06e-06 de diferencia):
https://github.com/santiquiroz/port-whisper-words-onnx

Corre UNA vez por trozo, sobre los tokens ya decodificados. Transcribir sigue
usando el decoder con cache, que es mucho mas rapido; este necesita la matriz de
atencion sobre toda la secuencia, que es justo lo que el cache evita recalcular.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Cada cuadro del encoder de Whisper cubre 20 ms: 30 s en 1500 cuadros.
SECONDS_PER_FRAME = 0.02

# El ancho que usa OpenAI. Impar a proposito: una mediana de ancho par no tiene
# elemento central y habria que promediar los dos, corriendo medio cuadro.
MEDIAN_FILTER_WIDTH = 7

# La atencion al ENCODER es la que mira el tiempo. La de `self_attn` mira los
# tokens ya emitidos y alinearia la letra contra si misma.
CROSS_ATTENTION_MARKER = "encoder_attn"

ATTENTION_GRAPH_NAME = "decoder_with_attentions.onnx"


@dataclass(frozen=True, slots=True)
class WordTiming:
    word: str
    start: float
    end: float


def cross_attention_tensor_names(model: Any) -> list[str]:
    """Los tensores de atencion cruzada, ORDENADOS POR CAPA.

    Ordenados importa: las alignment heads se indexan por capa, y ONNX no
    garantiza ningun orden de nodos. Tomarlos como salen daria cabezas de la capa
    equivocada, que no falla: solo alinea peor, que es la clase de error que no
    se descubre mirando.
    """
    nombres = [
        nodo.output[0]
        for nodo in model.graph.node
        if nodo.op_type == "Softmax" and CROSS_ATTENTION_MARKER in nodo.name
    ]

    def numero_de_capa(nombre: str) -> int:
        return int(nombre.split("layers.")[1].split("/")[0])

    return sorted(nombres, key=numero_de_capa)


def ensure_attention_graph(decoder_path: Path) -> tuple[Path, list[str]]:
    """El decoder con las atenciones publicadas, creandolo si hace falta.

    Se guarda al lado del original y se reusa: la cirugia es barata pero copia el
    grafo entero, y hacerla en cada trabajo escribiria cientos de MB por corrida.
    """
    import onnx

    destino = decoder_path.with_name(ATTENTION_GRAPH_NAME)
    if destino.exists():
        return destino, cross_attention_tensor_names(
            onnx.load(str(destino), load_external_data=False)
        )

    modelo = onnx.load(str(decoder_path))
    tensores = cross_attention_tensor_names(modelo)
    if not tensores:
        raise RuntimeError(
            f"El decoder {decoder_path.name} no expone atenciones cruzadas ni por "
            "dentro. Suele pasar cuando el export fusiono la atencion, y ahi la "
            "matriz de pesos no existe como tensor."
        )
    ya = {salida.name for salida in modelo.graph.output}
    for nombre in tensores:
        if nombre not in ya:
            modelo.graph.output.append(onnx.ValueInfoProto(name=nombre))
    onnx.save(modelo, str(destino))
    return destino, tensores


def median_filter(matrix: np.ndarray, width: int = MEDIAN_FILTER_WIDTH) -> np.ndarray:
    """Mediana movil sobre el ultimo eje, con los bordes replicados.

    Replicados y no rellenados con cero: un cero al borde es una atencion que no
    se midio, y arrastraria la mediana de los primeros cuadros hacia abajo, que
    es justo donde suele empezar la primera palabra.
    """
    if width <= 1 or matrix.shape[-1] < width:
        return matrix
    pad = width // 2
    acolchado = np.pad(matrix, [(0, 0)] * (matrix.ndim - 1) + [(pad, pad)], mode="edge")
    ventanas = np.lib.stride_tricks.sliding_window_view(acolchado, width, axis=-1)
    return np.median(ventanas, axis=-1)


def attention_to_cost(
    cross_attentions: np.ndarray, alignment_heads: list[tuple[int, int]]
) -> np.ndarray:
    """De [capas, cabezas, tokens, cuadros] a una matriz de costo [tokens, cuadros]."""
    if not alignment_heads:
        raise ValueError(
            "Sin alignment heads no se puede alinear: promediar todas las cabezas "
            "mete las que atienden a contenido y corre las marcas."
        )
    elegidas = np.stack(
        [cross_attentions[capa, cabeza] for capa, cabeza in alignment_heads], axis=0
    )
    media = elegidas.mean(axis=-1, keepdims=True)
    desvio = elegidas.std(axis=-1, keepdims=True)
    normalizadas = (elegidas - media) / np.maximum(desvio, 1e-8)
    return -median_filter(normalizadas).mean(axis=0)


def dtw(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camino monotono de costo minimo: (indices de token, indices de cuadro).

    Monotono porque el habla avanza; permitir retroceso daria palabras que
    empiezan antes que la anterior.
    """
    n_tokens, n_frames = cost.shape
    acumulado = np.full((n_tokens + 1, n_frames + 1), np.inf, dtype=np.float64)
    acumulado[0, 0] = 0.0
    origen = np.zeros((n_tokens + 1, n_frames + 1), dtype=np.uint8)

    for i in range(1, n_tokens + 1):
        fila_previa = acumulado[i - 1]
        fila = acumulado[i]
        for j in range(1, n_frames + 1):
            candidatos = (fila_previa[j - 1], fila_previa[j], fila[j - 1])
            mejor = int(np.argmin(candidatos))
            fila[j] = cost[i - 1, j - 1] + candidatos[mejor]
            origen[i, j] = mejor

    tokens: list[int] = []
    frames: list[int] = []
    i, j = n_tokens, n_frames
    while i > 0 and j > 0:
        tokens.append(i - 1)
        frames.append(j - 1)
        paso = origen[i, j]
        if paso == 0:
            i, j = i - 1, j - 1
        elif paso == 1:
            i -= 1
        else:
            j -= 1
    return np.array(tokens[::-1]), np.array(frames[::-1])


def group_tokens_into_words(pieces: list[str]) -> tuple[list[str], list[list[int]]]:
    """Junta los tokens de BPE en palabras.

    El tokenizador parte una palabra en varios tokens y marca el comienzo con un
    espacio al principio. Alinear por token y mostrar eso seria iluminar pedazos
    de palabra.
    """
    palabras: list[str] = []
    grupos: list[list[int]] = []
    for indice, pieza in enumerate(pieces):
        arranca = pieza.startswith(" ") or not palabras
        if arranca:
            palabras.append(pieza.strip())
            grupos.append([indice])
        else:
            palabras[-1] += pieza
            grupos[-1].append(indice)
    return palabras, grupos


def words_from_alignment(
    cost: np.ndarray,
    words: list[str],
    word_token_groups: list[list[int]],
    *,
    time_offset: float = 0.0,
) -> list[WordTiming]:
    if len(words) != len(word_token_groups):
        raise ValueError(
            f"Hay {len(words)} palabras y {len(word_token_groups)} grupos: alinear "
            "con grupos que no corresponden da marcas de otra palabra."
        )
    indices_token, indices_cuadro = dtw(cost)
    tiempos: list[WordTiming] = []
    for palabra, grupo in zip(words, word_token_groups):
        pertenece = np.isin(indices_token, grupo)
        if not pertenece.any():
            ultimo = tiempos[-1].end if tiempos else time_offset
            tiempos.append(WordTiming(word=palabra, start=ultimo, end=ultimo))
            continue
        cuadros = indices_cuadro[pertenece]
        tiempos.append(
            WordTiming(
                word=palabra,
                start=time_offset + float(cuadros.min()) * SECONDS_PER_FRAME,
                end=time_offset + float(cuadros.max() + 1) * SECONDS_PER_FRAME,
            )
        )
    return tiempos


def parse_alignment_heads(raw: Any) -> list[tuple[int, int]]:
    """`[[capa, cabeza], ...]` de la config del generador."""
    if not raw:
        return []
    return [(int(capa), int(cabeza)) for capa, cabeza in raw]


def align_words(
    *,
    model: Any,
    processor: Any,
    features: Any,
    tokens: Any,
    model_dir: Path,
    audio_seconds: float | None = None,
) -> list[WordTiming]:
    """Los tiempos de cada palabra de UN trozo de 30 s.

    Corre el encoder y el decoder-con-atenciones por separado en vez de reusar lo
    que hizo `generate`: optimum no devuelve los estados del encoder ni las
    atenciones, y volver a correr el encoder cuesta bastante menos que decodificar
    de nuevo.
    """
    import onnxruntime as ort

    cabezas = parse_alignment_heads(
        getattr(model.generation_config, "alignment_heads", None)
    )
    if not cabezas:
        # Sin cabezas declaradas no se puede alinear bien, y alinear mal es peor
        # que no resaltar: el karaoke queda corrido y parece un bug del video.
        return []

    decoder_path = model_dir / "onnx" / "decoder_model.onnx"
    if not decoder_path.exists():
        return []
    grafo, nombres = ensure_attention_graph(decoder_path)

    # `attention_mask` es posicional obligatorio en optimum aunque el encoder de
    # Whisper no lo use: la entrada siempre son 30 s acolchados, sin partes
    # variables que enmascarar.
    encoder_out = model.encoder(features, None)
    hidden = _to_numpy(encoder_out.last_hidden_state)
    ids = _to_numpy(tokens).astype(np.int64)
    if ids.ndim == 1:
        ids = ids[None, :]

    sesion = ort.InferenceSession(str(grafo), providers=["CPUExecutionProvider"])
    salidas = sesion.run(nombres, {"input_ids": ids, "encoder_hidden_states": hidden})
    # ONNX pliega batch y cabezas en un solo eje; el alineador indexa
    # [capa][cabeza], asi que se separan de nuevo.
    cross = np.stack(salidas, axis=0)
    if cross.ndim == 4:
        cross = cross[:, None]
    cross = cross[:, 0]

    piezas = _token_pieces(processor, ids[0])
    palabras, grupos = group_tokens_into_words(piezas)
    if not palabras:
        return []
    # Recortar al audio REAL antes de alinear. El encoder siempre devuelve 1500
    # cuadros porque acolcha la entrada a 30 s, y dejar el relleno adentro hace
    # que el DTW —que esta obligado a consumir todos los cuadros— estire las
    # ultimas palabras hasta el final del acolchado. El sintoma es todas las
    # marcas apiladas cerca del segundo 30 con un audio de 20.
    cross = _trim_to_audio(cross, audio_seconds)
    costo = attention_to_cost(cross, cabezas)
    return words_from_alignment(costo, palabras, grupos)


def _trim_to_audio(cross: np.ndarray, audio_seconds: float | None) -> np.ndarray:
    if audio_seconds is None or audio_seconds <= 0:
        return cross
    cuadros = int(round(audio_seconds / SECONDS_PER_FRAME))
    if cuadros <= 0 or cuadros >= cross.shape[-1]:
        return cross
    return cross[..., :cuadros]


def _to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _token_pieces(processor: Any, ids: np.ndarray) -> list[str]:
    """El texto de cada token, sin los especiales.

    Sin los especiales porque `<|startoftranscript|>` y las marcas de tiempo no
    son palabras: dejarlas dentro las convertiria en "palabras" que el karaoke
    intentaria resaltar.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    piezas: list[str] = []
    especiales = set(getattr(tokenizer, "all_special_ids", []) or [])
    for token_id in ids.tolist():
        if token_id in especiales:
            continue
        texto = tokenizer.decode([token_id], skip_special_tokens=True)
        # Las marcas de tiempo decodifican a vacio: descartarlas evita palabras
        # fantasma sin texto.
        if texto:
            piezas.append(texto)
    return piezas


def attach_words_to_segments(segments: list, words: list[WordTiming]) -> list:
    """Reparte las palabras en los segmentos que las contienen.

    Por el punto MEDIO de la palabra y no por su inicio: una palabra que arranca
    justo en el limite entre dos lineas caeria en la anterior por un centesimo, y
    ahi se resalta en la linea equivocada.
    """
    from app.services.subtitles import WordSpan

    if not words:
        return segments
    resultado = []
    for segmento in segments:
        propias = tuple(
            WordSpan(word=p.word, start=p.start, end=p.end)
            for p in words
            if segmento.start <= (p.start + p.end) / 2 <= segmento.end
        )
        resultado.append(
            type(segmento)(
                start=segmento.start,
                end=segmento.end,
                text=segmento.text,
                words=propias,
            )
        )
    return resultado
