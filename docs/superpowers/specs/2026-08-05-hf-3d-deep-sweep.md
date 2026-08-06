# Barrido profundo de Hugging Face: qué se puede correr de verdad (2026-08-05)

Seis ángulos de búsqueda en paralelo sobre el Hub, más una pasada adversarial que
intentó **tumbar** cada candidato leyendo el `LICENSE` crudo, el `requirements.txt`
y el `config.json`. No rankings, no artículos de blog.

Esto corrige el veredicto del mismo día que decía que no existía un motor 3D
usable con licencia limpia y sin CUDA. **Existen varios.**

## El hallazgo que más importa para piezas que tienen que encajar

**`Max2475/Qwen3.5-9B-OpenSCAD-Instruct`** — Apache-2.0.

Es el único camino encontrado que da **cotas exactas desde una descripción**, que
es justo lo que toda la investigación anterior daba por imposible:

- Emite **código OpenSCAD**: CSG paramétrico, con medidas de verdad, no una malla.
- Exporta STL nativo.
- Se distribuye **únicamente como GGUF** (`Q4_K_M`, ~8,95 GB) → llama.cpp con
  backend **Vulkan** sobre la RX 7800 XT, sin compilar nada. **Cero PyTorch, cero
  CUDA.** Es el mismo backend Vulkan que Upflow ya usa para ncnn y RIFE.

Dicho de otra forma: la frase "la IA no te va a dar una pieza acotada" era falsa.
Lo que no la da es un generador de **malla**. Un modelo que escribe **código CAD**
sí, porque las cotas están en el código.

## Lo demás que sobrevivió

| Modelo | Licencia | Salida | Sin CUDA |
|---|---|---|---|
| **Shap-E** (OpenAI) | MIT | malla | verificado y **ya integrado** |
| **PartCrafter** | MIT (Hub y repo) | malla **separada en partes** | verificado por grep sobre el repo |
| **CAD-Coder** (`gudo7208`, Qwen2.5-7B) | Apache-2.0 | código CadQuery | probable, sin ejecutar |
| **CADReasoner** | Apache-2.0 | código CadQuery, refinado por realimentación geométrica | parcial |
| **CADFS-2B** | MIT | FeatureScript de Onshape | el modelo sí; el formato ata a Onshape |
| **TripoSR ONNX** (varios exports) | MIT | malla | verificado leyendo los `.onnx` |
| **FreeSplatter** vía `free-splatter.cpp` | Apache-2.0 | gaussianas | verificado, motor C/C++ |

PartCrafter merece atención aparte: genera el objeto **ya separado en partes**,
que para imprimir es directamente útil — cada parte se orienta y se imprime por
separado en vez de pelear con voladizos de un objeto entero.

## La trampa que atrapó la pasada adversarial

**`CAD-Coder` de `anniedoris`/LLaVA**: el repo de GitHub dice Apache-2.0, pero los
pesos derivan de `lmsys/vicuna-13b-v1.5`, que arrastra la **Llama 2 Community
License**. Descartado.

Es exactamente el mismo patrón que ya había aparecido con TRELLIS: la licencia del
repo puede ser limpia y el modelo estar igual bloqueado por lo que hereda o por lo
que depende. Mirar solo el tag del Hub no alcanza.

## Qué queda pendiente

El carril OpenSCAD. Necesita:

1. `llama.cpp` con backend Vulkan como pack vendorizado (precedente exacto: ffmpeg,
   RIFE y Magpie ya viajan así).
2. El GGUF de 8,95 GB como descarga opcional.
3. Un intérprete de OpenSCAD para pasar de código a STL — OpenSCAD tiene binario
   de línea de comandos, GPL-2.0, que se ejecuta como proceso aparte igual que
   Magpie.
4. Y lo de siempre: lo que salga pasa por el banco antes de llegar al usuario.

Es la pieza más valiosa que queda del módulo, y la única que cierra el círculo
entre "describilo con palabras" y "esto mide 80 mm".
