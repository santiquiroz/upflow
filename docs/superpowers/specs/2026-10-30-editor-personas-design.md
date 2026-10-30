# Editor: herramienta Personas (elegir, pulir, revisar)

Fecha: 2026-10-30. Rama: `feature/borrado-pulido` (worktree `C:/wt/upflow-borrado`).

## Por qué

El dueño editó 60 fotos de grado (`C:\Users\santi\Desktop\imagenesThomas`) para un álbum impreso:
quitar a toda la gente del fondo y dejar a los sujetos del frente. Hoy en el Editor eso es tocar a
cada persona con "Tocar objeto", una por una, y proteger a los sujetos con "Proteger"; en una foto
con 10 personas son 10-15 clics, y el que se escapa queda en la foto. El trabajo de esta sesión
mejoró el motor (OSOR para huecos grandes, mate por color en el borde del sujeto, guarda de caras,
regla de superposición en el Editor), pero nada de eso se ve en la interfaz: el usuario no sabe
qué motor usó cada zona, ni si quedó alguien, ni puede rehacer una zona sola.

Pedido del dueño, en sus palabras: "funciones de reconocimiento de rostros, cuerpos y así en el
front del editor, poder seleccionar a quién quiero eliminar, a quién no"; la lista a la derecha
(como la opción A de la maqueta); "al elegirlos y darle aceptar, que agregue al contorno a eliminar
solo a la persona, luego pulirlo con el pincel quitando y agregando cosas que de pronto no
detectaste".

## Objetivos

1. **Elegir**: al abrir una foto en el Editor, Upflow detecta a todas las personas, propone quiénes
   son sujetos (Conservar) y quiénes fondo (Quitar), y el usuario corrige en una lista a la derecha.
2. **Pulir**: "Aceptar" pone en la máscara de borrado solo las siluetas marcadas para quitar, y
   protege las que se conservan; después se refina con las herramientas de siempre.
3. **Revisar**: el resultado muestra, por zona, qué motor la llenó y si quedó una persona; cada
   zona se puede rehacer sola con otro motor o describiendo el fondo.
4. Cerrar los dos límites de la sesión: (a) fantasmas de multitud sin cara frontal que el detector
   de caras no ve (2272, 2222, 2077); (b) la API no devuelve qué motor usó cada zona.
5. Que todo esto llegue al Upflow instalado: paquetes de modelos descargables por el instalador.

## No objetivos

- Reconocer identidades (saber que la "Graduada 1" es la misma persona en otra foto).
- Procesar lotes de fotos desde la UI (el lote de esta sesión fue un script; no se productiza acá).
- Rediseñar el resto de Upflow o su tema.
- Video.

## Anti-objetivos (fallaría aunque "funcione")

- Borrar a alguien sin que el usuario lo haya visto en rojo y aceptado: la detección solo propone.
- Que el Editor tarde más en abrir una foto: la detección corre en segundo plano y las demás
  herramientas se pueden usar mientras tanto.
- Que un sujeto quede marcado como fondo sin que se note: la lista siempre muestra a todos.
- Romper las herramientas actuales (Tocar objeto, Proteger, Pincel, Goma, Lazo, Rectángulo) o su
  historial de deshacer.

## Lo que decidió el dueño

- Lista de personas a la derecha, con la cara de cada una y Conservar / Quitar (opción A).
- "Aceptar" agrega al contorno de borrado solo la silueta de las personas a quitar.
- Después se pule con pincel y goma.
- Aprobó el flujo de tres pasos de la maqueta (elegir, pulir, revisar), incluido el paso Revisar que
  propuse yo.
- **El Editor entero pasa a tres columnas** (maqueta `editor-tres-columnas.html`):
  izquierda un riel vertical con todas las herramientas (Personas, Tocar objeto, Proteger, Pincel,
  Goma, Lazo, Rectángulo, Mover vista y abajo zoom −/+, Ajustar, Deshacer, Limpiar); centro la foto;
  derecha un panel contextual que muestra **Personas** (la lista) cuando esa herramienta está activa,
  **Acción** el resto del tiempo (Quitar / Reemplazar, Motor, Dispositivo, Opciones avanzadas e
  Insertar objeto plegados, botón "Quitar selección" / "Reemplazar selección"), y **Revisar zonas**
  cuando hay un resultado (con "Seguir editando este resultado" y "Usar otra imagen"). La
  herramienta activa se marca con fondo `--surface-2` y un borde ámbar a la izquierda, no con relleno
  ámbar: el único ámbar lleno de la pantalla es el botón de acción del panel.
- **El borrado automático no necesita SDXL**: un motor "Borrado automático (LaMa + OSOR)" que aparece
  con solo LaMa instalado y es el predeterminado en Quitar. SDXL queda para Reemplazar y para
  "Describir el fondo…".

Maquetas aprobadas en `C:\personal\upflow-mockups\2026-10-30-personas\`: `personas-flujo-v3.html` (los
tres pasos) y `editor-tres-columnas.html` (el Editor entero), con sus generadores `mock_flujo.py` y
`mock_layout.py`. Son fragmentos para el visor de brainstorming (las imágenes `/files/*.jpg` están en
la misma carpeta). Quedan fuera del repo porque usan una foto real de personas.

## Comportamiento

### Paso 1: Elegir

- Herramienta nueva **Personas** en el riel izquierdo del Editor, primera de la lista, con ícono
  (estilo Lucide, como el resto) y texto.
- Al cargar una imagen en el Editor se pide la detección en segundo plano. Mientras corre, el panel
  muestra un esqueleto ("Buscando personas…"); al terminar, la herramienta queda lista. Si no hay
  personas: estado vacío con texto ("No encontré personas en esta foto") y las demás herramientas
  siguen igual.
- Panel derecho "Personas en la foto" con dos grupos, **Al frente** y **Al fondo**, cada uno con su
  acción de grupo ("Quitar a todos" / "Conservar a todos"). Cada fila: miniatura de la cara (o de
  la cabeza si no hay cara), un nombre provisional (Persona 1, 2… ordenadas de izquierda a derecha)
  y un selector de dos estados Conservar (verde, ícono check) / Quitar (rojo, ícono x).
- Sobre la foto, cada persona detectada tiene un contorno punteado: verde si Conservar, rojo si
  Quitar.
- Vínculo lista ↔ foto: pasar el mouse por una fila resalta a esa persona en la foto (contorno
  ámbar y etiqueta con el nombre); pasar el mouse por una persona en la foto resalta su fila; un
  clic sobre la persona en la foto cambia su estado. Todo accesible por teclado (filas enfocables,
  flechas para moverse, espacio para cambiar estado).
- Casilla "Proteger a las que conservo", marcada por defecto.
- Botón principal (ámbar, uno solo por pantalla): "Agregar N siluetas al borrado", donde N es la
  cantidad en Quitar; deshabilitado si N = 0.

### Paso 2: Pulir

- "Agregar" pide la silueta de cada persona (ver Backend) y la suma al historial del Editor como
  entradas normales: las de Quitar como `segment` (rojo) y, con la casilla marcada, las de Conservar
  como `protect` (verde). Cada entrada guarda su punto (el centro de la cara o, si no hay cara, el
  centro del cuarto superior de la caja), así la regla de superposición de 44a58ac funciona igual
  que con los toques: el borrado gana sobre una protección que se pasó de largo, salvo que el
  segmento de borrado contenga el punto de esa protección.
- Las entradas se agregan como un solo paso de deshacer (un Ctrl+Z las quita todas).
- La herramienta activa pasa a **Pincel**. Una pista sobre la foto: "Pincel suma · Goma resta ·
  [ ] tamaño · Ctrl+Z deshacer".
- El panel pasa a un resumen: "N siluetas en el borrado", chips con las caras, el texto de cómo
  pulir, "Volver a elegir" (vuelve a la lista con el estado anterior; las siluetas ya agregadas se
  quitan del historial) y el botón principal de siempre "Quitar selección".

### Paso 3: Revisar

- Después de un borrado con estrategia automática, el panel derecho muestra **Revisar zonas**: una
  fila por zona (grupo de `auto_erase`) con número, quién estaba ahí (nombres de las personas
  cuyas siluetas caen en la zona; si no hay, "Zona N"), el motor (LaMa, OSOR, LaMa + OSOR, SDXL,
  Híbrido) y el estado: "Sin personas" (check) o "Se vio una persona: lo repitió solo" (alerta),
  o "Todavía se ve alguien" (alerta, cuando la repetición tampoco alcanzó).
- Sobre el resultado, cada zona tiene su número. Pasar el mouse por una fila o un número muestra
  esa zona antes / después (recorte del original contra el resultado).
- **Rehacer** por zona: menú con LaMa, OSOR, LaMa + OSOR y "Describir el fondo…" (abre un campo de
  texto; usa SDXL con ese texto como prompt). Rehacer corre un trabajo nuevo con el resultado actual
  como base y solo esa zona como máscara; al terminar reemplaza el resultado y actualiza la fila.

## Backend

### Detector de personas (mi decisión)

- **YOLOX-s en ONNX** (Megvii, Apache-2.0; release oficial `yolox_s.onnx`), clase `person` de COCO,
  entrada 640. Upflow es MIT: no sirve Ultralytics YOLO (AGPL). Corre por el mismo `ep_registry`
  que el resto (nativo → DirectML → CPU). Si al medir (ver Criterios) no alcanza, el builder puede
  cambiarlo por RT-DETR (Apache-2.0) sin cambiar el resto del diseño.
- Paquete nuevo `person-detector` con su entrada en el instalador de modelos y en
  `missing_pack.py`. Sin el paquete, la herramienta Personas aparece deshabilitada con el motivo
  (mismo patrón que "Tocar objeto" sin MobileSAM).

### Endpoints

- `POST /api/v1/editor/people` `{imageToken}` → `{people: [{id, box:[l,t,r,b], score, role:
  "subject"|"background", face:[l,t,r,b] | null}]}` en coordenadas de la imagen original. Cachea por
  imagen como los embeddings de MobileSAM.
- `POST /api/v1/editor/segment` acepta además `box:[l,t,r,b]` (excluyente con `x,y`). El decoder de
  MobileSAM recibe la caja como dos puntos con labels 2 y 3 (convención SAM). Devuelve la misma PNG
  de máscara que hoy.
- La miniatura de la cara la recorta el frontend de la imagen que ya tiene (no hace falta endpoint).

### Frente o fondo (mi decisión, a medir)

Puntaje por persona con tres señales: altura de la caja relativa a la imagen, nitidez (varianza del
laplaciano dentro de la caja contra la de toda la foto) y cercanía al centro horizontal. Sujeto = las
de puntaje alto y relativamente cerca del máximo de la foto. Umbrales a calibrar con las 60 fotos
(ver Criterios). La cara (`face`) sale de Haar (OpenCV, ya instalado) dentro de la mitad superior de
cada caja; si no hay, `null`.

### Guarda de personas después del borrado (límite a)

En `auto_erase`, después de llenar un grupo grande con OSOR: además de la guarda de caras actual
(`fill_faces.faces_inside`, Haar a 0.5 y 0.25), correr el detector de personas sobre el recorte del
grupo ya llenado; si aparece una persona con puntaje alto cuya caja cae mayormente dentro de lo
borrado, se trata igual que una cara: se repite con LaMa primero y OSOR encima. Si después de
repetir sigue apareciendo, la zona se marca `residualPerson: true` (no hay más reintentos).

### Zonas en la respuesta del trabajo (límite b)

- `GET /api/v1/generation/jobs/{id}` agrega `eraseZones: [{index, box:[l,t,r,b], engine, guard:
  "face"|"person"|null, residualPerson}]` a partir de lo que ya guarda `auto_erase` en
  `request.erase_auto` (hoy queda en `job.metadata["eraseAuto"]` pero no se serializa). `box` es la
  caja del recorte del grupo (las cajas de los grupos no se solapan: `edit_groups` las fusiona).
- Parámetro nuevo del trabajo `eraseEngine: "lama"|"osor"|"lama+osor"|"sdxl"` que fuerza el motor
  en todos los grupos (para Rehacer). Sin él, la estrategia automática de siempre.

### Borrado automático sin SDXL (decisión del dueño)

- Hoy la estrategia automática solo se activa con un modelo SDXL de relleno elegido y LaMa instalado
  (`shouldUseAutoErase` en el frontend) y el trabajo carga el pipeline SDXL aunque todas las zonas
  terminen en LaMa u OSOR. Pasa a ser un motor propio, "Borrado automático (LaMa + OSOR)", listado
  como modelo de borrado cuando LaMa está instalado, sin validarse como modelo SDXL.
- Reparto por zona: chica → LaMa; grande → OSOR si su paquete está instalado; si no hay OSOR, SDXL
  solo si hay un modelo SDXL de relleno instalado (la elección de hoy entre híbrido y directo); si
  tampoco, LaMa. SDXL se carga recién cuando una zona lo necesita, nunca de antemano.
- "Describir el fondo…" en Rehacer y el modo Reemplazar quedan deshabilitados, con el motivo visible,
  si no hay un modelo SDXL de relleno instalado.

## Distribución (que llegue al Upflow instalado)

- Paquete `person-detector`: descarga de `yolox_s.onnx` desde el release oficial, con hash fijo
  (mismo patrón que `test_lama_download_pin.py`).
- Paquete `osor`: hoy solo lo arma `scripts/build_osor_pack.py` desde fuentes locales del spike. El
  instalador debe poder: descargar de Hugging Face la base `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
  (unet y vae fp16, scheduler, y los dos text encoders con sus tokenizers solo para calcular los
  embeddings del prompt fijo "Remove the instance of object"), los pesos `QinmingZhou/OSOR`
  (`sdxlinpainting_phase2.pth`), calcular `prompt_embeds.pt` y correr el armado (fusión del LoRA).
  Son ~9 GB de descarga; el paquete final ~5 GB. Licencias: código OSOR MIT, pesos y base CreativeML
  Open RAIL++-M (el NOTICE ya existe).
- Mergear la rama y publicar un release son decisiones del dueño; el plan no hace push ni merge sin
  que lo pida.

## Restricciones

- Tema: el de Upflow (grafito `--bg #0e0f13`, superficies `--surface`/`--surface-2`, acento ámbar
  `--accent #f5a524`, `--ok`, `--danger`); verde/rojo de los overlays del Editor como hoy. Íconos SVG
  estilo Lucide, nunca emoji. Color siempre con ícono o texto. Foco visible. Botones de al menos 36 px
  de alto. Respetar `prefers-reduced-motion`.
- Textos en `i18n/es.ts` y `i18n/en.ts`, con el voseo que ya usa la app ("Tocá", "Elegí").
- Modelos que corran en esta máquina (RX 7800 XT, DirectML) y en CPU.
- Código: funciones atómicas, complejidad baja, sin docstrings, comentario de una línea solo para un
  porqué no obvio. Tests nuevos para lo nuevo (pytest en backend, vitest en el Editor).

## Criterios de aceptación (medidos, no supuestos)

Datos para medir:
- Las 60 fotos: `C:\Users\santi\Desktop\imagenesThomas\_DSC*.jpg`.
- Clics de la sesión como verdad: `...\scratchpad\thomas\specs\all8.json` y
  `...\scratchpad\thomas\specs\fix9.json` (`protect` = clics sobre sujetos, `bg` = clics sobre gente
  de fondo; coordenadas de la foto original).
- Resultados con fallos conocidos: `...\scratchpad\thomas\out8\` y `...\scratchpad\thomas\out9\`
  (`_DSCnnnn_editada.jpg`, `mask_i.png` por grupo, log `all8.log` / `fix9.log`). Grupos con persona
  que quedó: v8 2145 g0, 2234 g2, 2268 g1, 2274 g0, 2199 g2, 2187 g0; v9 2145 g1, 2268 g1, 2274 g0,
  2272 g0, 2077 g0, 2222 g2, 2163 g2, 2177 g4. El resto de los grupos OSOR revisados quedaron limpios.

(`...\scratchpad` = `C:\Users\santi\AppData\Local\Temp\claude\c--personal-bipolar-code-bipolar-code\82a622fd-b78e-4d2c-b39d-f5c14ea7e944\scratchpad`; si ya no existe, el plan debe copiar estos datos a una carpeta propia antes de empezar.)

Cómo se cuenta (mi decisión; los clics no son cajas y una persona tiene varios clics):
- Objetivo = la máscara MobileSAM de cada clic (mismo punto, mismo modelo que en la sesión); los
  objetivos de una misma foto que se solapan > 50 % se funden en uno. Así un clic en el diploma o en
  un ramo es un objetivo más, no una persona.
- Un objetivo está "detectado" si alguna caja del detector cubre ≥ 50 % de su máscara. Los objetivos
  sin cobertura van a una hoja de contactos y se revisan a ojo una vez para separar personas perdidas
  (cuentan como fallo) de cosas que no son personas (se descartan). Esa lista revisada queda
  guardada y es la verdad fija para iteraciones siguientes.
- El rol esperado de un objetivo es el del clic que lo generó (`protect` → sujeto, `bg` → fondo); si
  un objetivo cae dentro de la caja de un sujeto más grande, cuenta para la caja más chica que lo
  cubre ≥ 50 %.
- "Grupos OSOR limpios" = todos los grupos con motor `osor` de `out8/all8.log` y `out9/fix9.log` que
  no están en la lista de fallos de arriba (todos fueron revisados a ojo en la sesión).

1. Detección: ≥ 95 % de los objetivos-persona detectados.
2. Frente/fondo: el rol propuesto coincide con el esperado en ≥ 90 % de los objetivos-persona, y
   ningún sujeto de primer plano grande sale propuesto como fondo en más de 2 fotos de 59.
3. Guarda de personas: sobre los 14 grupos con persona que quedó, la guarda (caras + personas) marca
   ≥ 12; sobre los grupos OSOR limpios, a lo sumo 2 falsas alarmas.
4. Tiempo: la detección de una foto 6000x4000 tarda < 3 s en esta máquina y no bloquea el Editor.
5. E2E por la UI (Playwright): abrir 2145, aceptar la propuesta, quitar; las zonas aparecen en
   Revisar con su motor; rehacer una zona con LaMa reemplaza solo esa zona.

## Decisiones menores (mías, para revisar)

- Máscara de Rehacer: la máscara de borrado original recortada a la caja de la zona (no la caja
  entera, que incluye el margen de contexto).
- Con `eraseEngine` forzado, la guarda de personas no cambia de motor: si queda alguien, la zona se
  marca `residualPerson` y el usuario decide.
- Antes de mandar cajas a MobileSAM, verificar las formas de entrada del decoder ONNX
  (`point_coords` admite N puntos; caja = esquinas con labels 2 y 3).
- Si personas muy chicas (balcones) se escapan del detector a 640, partir la foto en mosaico solo si
  entra en el presupuesto de 3 s.
- `peft` (lo usa `scripts/build_osor_pack.py` para fusionar el LoRA) se agrega a las dependencias del
  instalador del paquete OSOR; el runtime sigue sin necesitarlo.

## Lo que queda al builder

- Orden de las fases y cómo partir el trabajo (sugerencia: backend detector + endpoints, luego
  zonas y guarda, luego frontend, luego paquetes del instalador).
- Nombres internos, estructura de componentes React, cómo se dibuja el esqueleto de carga.
- Si la silueta por caja de MobileSAM sale mal en personas tapadas por otra, el builder puede sumar
  el punto de la cara como prompt positivo junto a la caja.
- Umbrales exactos (frente/fondo, puntaje del detector, solapamiento de la guarda), calibrados con
  los datos de arriba.
