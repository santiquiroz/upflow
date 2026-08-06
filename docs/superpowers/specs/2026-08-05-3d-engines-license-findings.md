# Motores 3D: qué se puede usar de verdad (verificado 2026-08-05)

> **ESTE DOCUMENTO QUEDÓ CORTO.** Un barrido posterior y más profundo sobre el Hub
> —seis ángulos en paralelo, leyendo el `LICENSE` crudo de cada repo— encontró
> varios motores usables que acá no aparecen, incluido uno que da **cotas exactas**
> desde una descripción. Ver
> [`2026-08-05-hf-3d-deep-sweep.md`](2026-08-05-hf-3d-deep-sweep.md).
>
> La conclusión de abajo sobre "esto no existe hoy" era producto de una búsqueda
> pobre, no de la realidad.

Verificación adversarial: a cada afirmación sobre licencia y hardware se le pidió
a un revisor independiente que la **refutara** leyendo la fuente primaria (el
`LICENSE` crudo del repo, la model card de Hugging Face, el código de arranque).
De seis afirmaciones, **cuatro salieron refutadas**. Ese es el punto: una licencia
mal leída es el error más caro que puede cometer este proyecto.

## El resultado, sin adornos

| Motor | Licencia propia | ¿Uso comercial? | ¿Sin CUDA? | Veredicto |
|---|---|---|---|---|
| **TripoSR** | MIT (código **y** pesos) | Sí, sin condiciones | Sí: `run.py` cae a CPU solo | **Único candidato limpio** |
| TRELLIS / TRELLIS.2 | MIT (código y pesos) | Sí, por su licencia | **No** | Bloqueado por dependencia, no por licencia |
| Stable Fast 3D | Stability AI Community | Sí, **si facturás ≤ US$1M/año** | Parcial | Usable con condición de ingresos |
| Hunyuan3D (toda la familia) | Tencent Hunyuan 3D Community | **No limpia** | — | Descartado |
| Pixal3D | — | — | No | Descartado |

## Las correcciones que importan

**TRELLIS no está bloqueado por su licencia.** Ese fue mi error inicial. El
`LICENSE` de `microsoft/TRELLIS.2` es MIT, copyright Microsoft, sin cláusula de
uso comercial ni de uso aceptable. Lo que lo bloquea es **de qué depende**:
kernels CUDA obligatorios y `nvdiffrast`, que sí es NVIDIA no comercial. La
distinción no es académica — significa que si algún día aparece un puerto sin
nvdiffrast, TRELLIS vuelve a la mesa. Hunyuan3D no vuelve: ahí el problema es la
licencia del modelo mismo.

**Stable Fast 3D permite uso comercial**, contra lo que decía la afirmación
original. La Stability AI Community License es *source-available*, no OSI, y
habilita uso comercial gratuito para personas u organizaciones con ingresos
anuales de hasta un millón de dólares. Para un taller es viable; queda como
segunda opción documentada, no descartada.

**TripoSR resistió el intento de refutación.** El revisor lo intentó y no pudo:
el `LICENSE` crudo del repo dice "MIT License, Copyright (c) 2024 Tripo AI &
Stability AI", y la model card lo etiqueta `license: mit`. Código y pesos, las
dos cosas.

## Lo que NO está verificado

El camino **sin CUDA de TripoSR que está realmente comprobado es PyTorch en CPU**,
porque `run.py` hace el fallback de primera parte. El bundle ONNX de terceros que
circula documenta una toolchain CUDA 12.4, y su README reclama Stability AI
Community License mientras su metadata dice `license: mit` — se contradice sola,
así que no se usa. Si el día de mañana se quiere DirectML, hay que exportar desde
el repo MIT con un script propio y **medir** qué porcentaje de nodos coloca en
`DmlExecutionProvider` antes de prometer nada.

## Por qué esto no cambia el plan

El hallazgo de mercado ya había dicho lo importante: **ningún generador de malla
da cotas ni aristas vivas**, y una pieza industrial o de carro necesita las dos.
Así que el generador nunca iba a ser el corazón del módulo. El corazón es el banco
de verificación —que ya funciona sobre cualquier STL venga de donde venga— y el
carril CAD paramétrico (`build123d`/`CadQuery` sobre OpenCASCADE, Apache-2.0, CPU
puro), que es lo único de toda la investigación que emite **STEP con cotas
exactas**.

Dicho derecho: si el pedido fuera "que la IA me genere una pieza de carro
acotada", la respuesta honesta hoy es que eso no existe con licencia limpia y sin
CUDA. Lo que sí existe es todo lo demás del camino.
