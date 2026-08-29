import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.services import model3d_service
from app.services.turnaround import Box


PANEL_BOXES = (
    Box(20, 20, 80, 120),
    Box(120, 28, 190, 126),
    Box(235, 15, 300, 110),
    Box(350, 25, 430, 117),
)


def _write_sheet(path: Path, boxes: tuple[Box, ...]) -> Path:
    width = max(box.x1 for box in boxes) + 20
    height = max(box.y1 for box in boxes) + 20
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle((box.x0, box.y0, box.x1 - 1, box.y1 - 1), fill="black")
    image.save(path)
    return path


@pytest.fixture
def fake_blender(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, dict[str, object]]] = []
    canned = {"ok": True, "source": "fake-blender"}

    def fake_run_script(_settings, script: str, payload: dict[str, object]):
        calls.append((script, payload))
        return canned

    monkeypatch.setattr(model3d_service.blender_service, "run_script", fake_run_script)
    return calls, canned


def test_audit_mesh_calls_blender_audit_script_and_returns_result(
    tmp_path: Path, fake_blender
) -> None:
    mesh = tmp_path / "character mesh.obj"
    calls, canned = fake_blender

    result = model3d_service.audit_mesh(object(), mesh)

    assert result is canned
    assert calls == [("audit_mesh.py", {"mesh": str(mesh)})]


def test_split_views_writes_named_tight_crops_with_crop_coordinate_ink_boxes(
    tmp_path: Path,
) -> None:
    boxes = PANEL_BOXES[:3]
    sheet = _write_sheet(tmp_path / "turnaround.png", boxes)
    out_dir = tmp_path / "missing" / "split"
    assert not out_dir.exists()

    views = model3d_service.split_views(sheet, out_dir)

    assert out_dir.is_dir()
    assert [view.name for view in views] == ["front", "side", "back"]
    assert [view.image.name for view in views] == ["front.png", "side.png", "back.png"]
    for view, sheet_box in zip(views, boxes):
        expected_path = out_dir / f"{view.name}.png"
        expected_crop_box = Box(0, 0, sheet_box.width, sheet_box.height)

        assert view.image == expected_path
        assert expected_path.is_file()
        assert view.ink == expected_crop_box
        assert view.ink != sheet_box
        with Image.open(expected_path) as crop:
            assert crop.size == (sheet_box.width, sheet_box.height)


def test_detected_view_as_payload_uses_json_serializable_ink_list(tmp_path: Path) -> None:
    view = model3d_service.DetectedView(
        name="front",
        image=tmp_path / "front.png",
        ink=Box(1, 2, 30, 40),
    )

    payload = view.as_payload()

    assert payload == {
        "image": str(tmp_path / "front.png"),
        "inkBox": [1, 2, 30, 40],
    }
    assert isinstance(payload["inkBox"], list)
    assert json.loads(json.dumps(payload)) == payload


def test_build_reference_scene_serializes_views_and_supports_height_override(
    tmp_path: Path, fake_blender
) -> None:
    views = [
        model3d_service.DetectedView("front", tmp_path / "front.png", Box(0, 0, 60, 100)),
        model3d_service.DetectedView("side", tmp_path / "side.png", Box(0, 0, 70, 98)),
    ]
    default_output = tmp_path / "default scene.blend"
    custom_output = tmp_path / "custom scene.blend"
    calls, canned = fake_blender

    default_result = model3d_service.build_reference_scene(object(), views, default_output)
    custom_result = model3d_service.build_reference_scene(
        object(), views, custom_output, height_meters=1.83
    )

    expected_views = {view.name: view.as_payload() for view in views}
    assert default_result is canned
    assert custom_result is canned
    assert calls == [
        (
            "build_reference_scene.py",
            {
                "views": expected_views,
                "heightMeters": 1.70,
                "output": str(default_output),
            },
        ),
        (
            "build_reference_scene.py",
            {
                "views": expected_views,
                "heightMeters": 1.83,
                "output": str(custom_output),
            },
        ),
    ]

def _hoja(tmp_path, tamano, bloques):
    from PIL import Image, ImageDraw

    imagen = Image.new("RGB", tamano, "white")
    dibujo = ImageDraw.Draw(imagen)
    for caja in bloques:
        dibujo.rectangle(caja, fill="black")
    destino = tmp_path / "hoja.png"
    imagen.save(destino)
    return destino


def test_una_hoja_sin_fondo_entre_vistas_nombra_esa_causa(tmp_path):
    # Un solo panel que ocupa casi toda la hoja no es "una vista": es que no
    # hubo por donde cortar. Culpar a vistas superpuestas manda a arreglar lo
    # que no esta roto.
    hoja = _hoja(tmp_path, (800, 400), [(10, 20, 790, 380)])

    avisos = model3d_service.sheet_warnings(hoja)

    assert len(avisos) == 1
    assert "fondo" in avisos[0]
    assert "superpuestas" not in avisos[0]


def test_mas_vistas_que_nombres_avisa_en_vez_de_descartarlas_callado(tmp_path):
    # zip() truncaba en silencio: en disco quedaban menos recortes que en la
    # hoja y la respuesta no lo mencionaba.
    bloques = [(20 + i * 145, 40, 130 + i * 145, 360) for i in range(6)]
    hoja = _hoja(tmp_path, (900, 400), bloques)

    avisos = model3d_service.sheet_warnings(hoja)

    assert len(avisos) == 1
    assert "6" in avisos[0] and "4" in avisos[0]


def _figura(tmp_path, nombre, ancho_por_banda):
    """Un monigote de bandas: cada banda es un rectangulo centrado."""
    from PIL import Image, ImageDraw

    alto = len(ancho_por_banda) * 10
    imagen = Image.new("RGB", (200, alto), "white")
    dibujo = ImageDraw.Draw(imagen)
    for i, ancho in enumerate(ancho_por_banda):
        y = i * 10
        dibujo.rectangle([100 - ancho // 2, y, 100 + ancho // 2, y + 10], fill="black")
    destino = tmp_path / nombre
    imagen.save(destino)
    return destino


def test_las_proporciones_marcan_dudosa_la_altura_en_la_que_las_vistas_no_coinciden(tmp_path):
    # Lo que las dos vistas ubican en el mismo lugar es una articulacion; lo que
    # cada una ve en otro lado es ruido. Promediarlo en silencio daria un numero
    # inventado con cara de medicion.
    cuello_arriba = [60, 60, 20, 60, 60, 60, 90, 60, 60, 60]
    cuello_igual_cadera_distinta = [60, 60, 20, 60, 60, 90, 60, 60, 60, 60]
    _figura(tmp_path, "front.png", cuello_arriba)
    _figura(tmp_path, "side.png", cuello_igual_cadera_distinta)

    medidas = model3d_service.measure_proportions(tmp_path, height_meters=1.70)

    por_nombre = {a["name"]: a for a in medidas["landmarks"]}
    assert por_nombre["cuello"]["agrees"] is True
    assert por_nombre["cadera"]["agrees"] is False
    assert "cadera" in medidas["uncertain"]
    assert por_nombre["cadera"]["disagreementCm"] > 0


def test_las_proporciones_necesitan_las_dos_vistas(tmp_path):
    _figura(tmp_path, "front.png", [60, 20, 60, 90])

    with pytest.raises(FileNotFoundError):
        model3d_service.measure_proportions(tmp_path)


def test_renombrar_vistas_intercambia_sin_pisar_archivos(tmp_path):
    # Intercambiar frente y espalda directamente pisaria un archivo con el otro,
    # asi que el renombrado pasa por nombres temporales.
    for nombre, ancho in (("front", 40), ("side", 20), ("back", 60)):
        _figura(tmp_path, f"{nombre}.png", [ancho] * 6)
    tamanos = {p.stem: p.stat().st_size for p in tmp_path.glob("*.png")}

    model3d_service.rename_views(tmp_path, ["back", "side", "front"])

    assert (tmp_path / "front.png").stat().st_size == tamanos["back"]
    assert (tmp_path / "back.png").stat().st_size == tamanos["front"]
    assert (tmp_path / "side.png").stat().st_size == tamanos["side"]


def test_renombrar_rechaza_un_nombre_que_la_escena_no_sabe_colocar(tmp_path):
    _figura(tmp_path, "front.png", [40] * 6)

    with pytest.raises(model3d_service.UnknownViewNameError, match="front"):
        model3d_service.rename_views(tmp_path, ["arriba"])


def test_renombrar_rechaza_nombres_repetidos(tmp_path):
    for nombre in ("front", "side"):
        _figura(tmp_path, f"{nombre}.png", [40] * 6)

    with pytest.raises(model3d_service.UnknownViewNameError, match="repetidos"):
        model3d_service.rename_views(tmp_path, ["front", "front"])


def test_renombrar_exige_un_nombre_por_vista(tmp_path):
    for nombre in ("front", "side"):
        _figura(tmp_path, f"{nombre}.png", [40] * 6)

    with pytest.raises(model3d_service.UnknownViewNameError, match="2 vistas"):
        model3d_service.rename_views(tmp_path, ["front"])


def test_remesh_manda_el_voxel_y_devuelve_las_dos_auditorias(tmp_path, monkeypatch):
    # Las dos auditorías viajan juntas porque un remesh gana topología y pierde
    # detalle: cuánto perdió solo se ve comparando las dos puntas.
    recibido = {}

    def fake_run(_settings, script, payload, **_kwargs):
        recibido["script"] = script
        recibido["payload"] = payload
        return {"mesh": payload["output"], "voxelMeters": payload["voxelMeters"],
                "before": {"faces": 288360}, "after": {"faces": 22568}}

    monkeypatch.setattr(model3d_service.blender_service, "run_script", fake_run)

    salida = model3d_service.remesh(
        object(), tmp_path / "entra.glb", tmp_path / "sale.glb", voxel_meters=0.03
    )

    assert recibido["script"] == "remesh.py"
    assert recibido["payload"]["voxelMeters"] == 0.03
    assert salida["before"]["faces"] > salida["after"]["faces"]


def _medida(promedio: float, *, ok: bool = True) -> dict[str, object]:
    return {"audit": {"ok": ok}, "fit": {"average": promedio, "views": []}}


def test_el_banco_ordena_por_calce(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    puntajes = {"floja": 0.30, "buena": 0.80, "media": 0.55}
    monkeypatch.setattr(
        model3d_service,
        "score_fit",
        lambda _s, ruta, *a, **k: _medida(puntajes[ruta.stem]),
    )

    tabla = model3d_service.compare_meshes(
        None,
        {nombre: tmp_path / f"{nombre}.glb" for nombre in puntajes},
        tmp_path,
        tmp_path,
        height_meters=1.0,
    )

    assert [r["name"] for r in tabla["ranked"]] == ["buena", "media", "floja"]
    assert tabla["winner"] == "buena"


def test_el_banco_no_corona_una_malla_rota_aunque_calce_mejor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Premiar una silueta linda sobre una malla rota es el falso positivo
    que este banco existe para no cometer."""
    resultados = {"rota": _medida(0.90, ok=False), "sana": _medida(0.60)}
    monkeypatch.setattr(
        model3d_service, "score_fit", lambda _s, ruta, *a, **k: resultados[ruta.stem]
    )

    tabla = model3d_service.compare_meshes(
        None,
        {nombre: tmp_path / f"{nombre}.glb" for nombre in resultados},
        tmp_path,
        tmp_path,
        height_meters=1.0,
    )

    assert [r["name"] for r in tabla["ranked"]] == ["rota", "sana"]
    assert tabla["winner"] == "sana"


def test_una_candidata_que_falla_no_tumba_el_banco(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Un motor que no arranca es lo que hay que VER en la tabla, no un stack
    trace que la deja vacia."""
    def _falla_una(_settings, ruta, *args, **kwargs):
        if ruta.stem == "explota":
            raise RuntimeError("el motor no arranco")
        return _medida(0.42)

    monkeypatch.setattr(model3d_service, "score_fit", _falla_una)

    tabla = model3d_service.compare_meshes(
        None,
        {"explota": tmp_path / "explota.glb", "anda": tmp_path / "anda.glb"},
        tmp_path,
        tmp_path,
        height_meters=1.0,
    )

    assert tabla["measured"] == 1
    assert tabla["failed"] == ["explota"]
    assert tabla["winner"] == "anda"
    assert "el motor no arranco" in tabla["ranked"][-1]["error"]


def test_el_banco_mide_cada_candidata_en_SU_eje(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """El eje va por candidata porque cada motor entrega en su propio marco.

    Un banco que impone una sola orientación a todas puntúa la ROTACIÓN de las
    que no coinciden. Medido: la misma malla de TripoSG dio 0.033 con el eje
    equivocado y 0.266 con el correcto, o sea que el eje decidía el ranking.
    """
    ejes: dict[str, str] = {}

    def _espia(_settings, ruta, _views, _render, **kwargs):
        ejes[ruta.stem] = kwargs["up_axis"]
        return _medida(0.5)

    monkeypatch.setattr(model3d_service, "score_fit", _espia)

    model3d_service.compare_meshes(
        None,
        [
            model3d_service.Candidata("blockout", tmp_path / "blockout.glb", "z_up"),
            model3d_service.Candidata("generada", tmp_path / "generada.glb", "y_up"),
        ],
        tmp_path,
        tmp_path,
        height_meters=1.0,
    )

    assert ejes == {"blockout": "z_up", "generada": "y_up"}


def test_el_banco_sigue_aceptando_un_diccionario_de_rutas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """La forma vieja no se rompe: todas heredan el eje del banco."""
    ejes: dict[str, str] = {}

    def _espia(_settings, ruta, _views, _render, **kwargs):
        ejes[ruta.stem] = kwargs["up_axis"]
        return _medida(0.5)

    monkeypatch.setattr(model3d_service, "score_fit", _espia)

    model3d_service.compare_meshes(
        None,
        {"una": tmp_path / "una.glb", "otra": tmp_path / "otra.glb"},
        tmp_path,
        tmp_path,
        height_meters=1.0,
        up_axis="y_up",
    )

    assert ejes == {"una": "y_up", "otra": "y_up"}


def test_la_huella_de_la_hoja_cambia_si_cambia_un_dibujo(tmp_path: Path) -> None:
    """Un puntaje sin su referencia no es reproducible.

    Costó una tarde el 2026-08-29: un proceso concurrente sobrescribió
    front.png a mitad de un barrido, las mallas seguían siendo byte a byte
    idénticas y la vista frontal caía de 0.54 a 0.06. La huella hace visible
    en un vistazo que cambió el dibujo y no el modelo.
    """
    vistas_dir = tmp_path / "vistas"
    vistas_dir.mkdir()
    for nombre in ("front", "side"):
        _write_sheet(vistas_dir / f"{nombre}.png", (Box(5, 5, 40, 60),))

    antes = model3d_service.sheet_digest(model3d_service.views_from_dir(vistas_dir))
    _write_sheet(vistas_dir / "front.png", (Box(5, 5, 30, 55),))
    despues = model3d_service.sheet_digest(model3d_service.views_from_dir(vistas_dir))

    assert antes != despues


def test_la_huella_no_depende_del_orden_de_lectura(tmp_path: Path) -> None:
    """Misma hoja, misma huella: si no, deja de servir para comparar corridas."""
    vistas_dir = tmp_path / "vistas"
    vistas_dir.mkdir()
    for nombre in ("front", "side", "back"):
        _write_sheet(vistas_dir / f"{nombre}.png", (Box(5, 5, 40, 60),))

    vistas = model3d_service.views_from_dir(vistas_dir)

    assert model3d_service.sheet_digest(vistas) == model3d_service.sheet_digest(vistas[::-1])
