# Cuatro pistas, y por qué son de un modelo peor

Upflow separa voz/instrumental desde hace varias versiones. Separar en **cuatro**
pistas —voz, batería, bajo y resto— era lo que faltaba, y no faltaba por falta de
modelo: faltaba por licencia.

## El problema no era técnico

El contrato de N stems entró en la v0.66.0 y quedó verificado con el modelo de 4
pistas de ZFTurbo: cuatro pistas separadas correctamente, todos los gates verdes,
y en fp16 corriendo a 1.23x tiempo real. **No se publicó ni se integró** porque
sus pesos no tienen licencia que permita redistribuirlos. El survey completo está
en el port; el resumen es que de todos los separadores de 4 pistas de calidad
alta, ninguno permite republicar los pesos:

| Candidato | Qué declara |
|---|---|
| BS-RoFormer 4 stems (ZFTurbo) | nada sobre los pesos; el MIT del repo cubre el código |
| Demucs `htdemucs` | "los pesos no están cubiertos por la licencia MIT, se proveen solo con fines científicos" |
| anvuew | GPL-3.0 |
| Sucial, becruily `deux` | CC BY-NC(-SA) |
| unwa/pcunwa, gabox (los de mayor SDR) | nada: ni licencia, ni model card |

El pedido a ZFTurbo está hecho
([MSST#249](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/249)),
pero un feature no puede esperar a que alguien conteste un issue.

## Lo que sí se puede publicar

**Open-Unmix `umxhq`**: MIT declarado en el
[record de Zenodo](https://zenodo.org/records/3370489), sobre los `.pth` mismos,
no sobre el código de otro repo. Es la diferencia entre "parece MIT" y "es MIT".

Portado en [port-openunmix-onnx](https://github.com/santiquiroz/port-openunmix-onnx),
con paridad de **109.8 a 135.5 dB SI-SDR** contra el separador oficial de upstream,
en CPU EP y en DirectML.

**Y es un modelo peor.** ~5.4 dB de SDR promedio contra ~9.4 de un RoFormer, y esa
diferencia se oye: deja más cruce entre pistas. La app lo dice en la advertencia
del modelo, antes de que el usuario lo elija, en lugar de dejar que lo descubra
con el resultado. Se ofrece por lo que **hace** —cuatro pistas— y no por cómo suena
comparado con el carril de voz.

Es rápido, eso sí: ~19x tiempo real con las cuatro pistas (unos 13 s por canción
de 4 minutos), contra el ~1.2x del RoFormer de una sola.

## Qué se tocó, y qué NO

Lo que **no** hubo que tocar es la parte interesante: el contrato de N stems de la
v0.66.0 funcionó sin un cambio. `stems_in_catalog_order` mapea las cuatro salidas
0..3 y no calcula residuo, porque un modelo que estima las cuatro pistas no deja
nada sin explicar. Ese contrato se escribió para un modelo que no existía todavía
y aguantó al primero real.

Lo que sí:

- **`SeparationModelSpec.files`** — hasta ahora un modelo era un archivo. umxhq son
  cuatro grafos que no sirven de a uno. La propiedad vive en la BASE y devuelve
  `(filename,)` por defecto, así que nada aguas abajo tiene que preguntar de qué
  tipo es la entrada.
- **`installed_model_ids` mira `files`** — con tres de cuatro grafos presentes, el
  picker ofrecía el modelo y el trabajo moría al cargar la segunda sesión.
- **`UmxSeparator._create_session` devuelve un dict de sesiones.** La base cachea lo
  que ese método devuelva sin mirarlo, así que el LRU sigue valiendo: un modelo
  vivo por dispositivo, con sus cuatro grafos adentro.
- **Sin chunking, a propósito.** El eje temporal del grafo es dinámico y la red es
  una LSTM: cada corte reiniciaría el estado de la recurrencia.
- **Las claves de i18n de batería/bajo/resto** — que el comentario de
  `separation_spec.py` decía explícitamente que no debían existir hasta que hubiera
  un modelo que las emitiera. Ahora lo hay.

## El detalle que casi se cuela

El script de descarga asumía un archivo por modelo. Al agregar el soporte de
varios, el test anti-drift del catálogo falló — correctamente: verificaba que la
URL completa estuviera en el `.ps1`, y ahora se arma con base + nombre. Se extendió
en vez de aflojarse, y ahora verifica **los cuatro archivos y los cuatro hashes**,
que es más de lo que verificaba antes: con la versión vieja, un `.ps1` que bajara
tres grafos buenos y uno corrupto pasaba la suite.
