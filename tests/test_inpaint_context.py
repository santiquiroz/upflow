"""Cuanto contexto ve el modelo alrededor de lo que se marco.

Reportado el 2026-08-06: al borrar un barbijo de un rostro, la boca generada
salia mirando en una direccion completamente distinta a la de la cara. Causa: el
contexto era FIJO, 48 px, sin importar el tamaño de lo marcado ni de la foto.

En una foto grande, una boca ocupa unos 300 px. Con 48 px alrededor, el modelo
recibe el agujero y un pedacito de menton — no ve la cara, asi que no tiene con
que saber hacia donde mira. Inventa una boca plausible en si misma y ajena a todo
lo demas.

Es el mismo error que la tolerancia fija de soldadura en el banco de impresion:
una constante absoluta aplicada a magnitudes que varian ordenes de magnitud.
"""

from __future__ import annotations

import pytest

from app.services.inpaint_mask import MIN_CONTEXT_PX, compute_crop_box, context_padding_for


class TestElContextoEscalaConLoMarcado:
    def test_una_mascara_grande_recibe_mas_contexto_que_una_chica(self) -> None:
        chica = context_padding_for((100, 100, 140, 140))
        grande = context_padding_for((100, 100, 400, 400))

        assert grande > chica

    def test_una_mascara_de_una_boca_ve_la_cara_entera(self) -> None:
        # Una boca de 300 px: el contexto tiene que alcanzar para que entren los
        # ojos y el contorno de la cabeza, que es lo que dice hacia donde mira.
        padding = context_padding_for((1000, 1200, 1300, 1400))

        assert padding >= 150, "con menos que esto el modelo no ve la cara"

    def test_una_mascara_diminuta_igual_recibe_un_minimo(self) -> None:
        # Una fraccion de algo diminuto seria casi nada, y sin nada alrededor no
        # hay con que continuar ni siquiera una textura.
        assert context_padding_for((10, 10, 12, 12)) == MIN_CONTEXT_PX

    def test_el_lado_mayor_manda(self) -> None:
        # Una mancha larga y finita necesita el contexto de su largo: mirar solo
        # el alto daria una franja sin nada a los costados.
        alargada = context_padding_for((0, 0, 400, 20))

        assert alargada == context_padding_for((0, 0, 400, 400))


class TestElRecorteSigueSiendoValido:
    def test_el_recorte_nunca_se_sale_de_la_foto(self) -> None:
        caja = compute_crop_box((10, 10, 300, 300), context_padding_for((10, 10, 300, 300)), (400, 400))
        izquierda, arriba, derecha, abajo = caja

        assert izquierda >= 0 and arriba >= 0
        assert derecha <= 400 and abajo <= 400

    def test_lo_marcado_entra_entero_en_el_recorte(self) -> None:
        # Si el recorte no contiene la marca, se edita otra cosa.
        bbox = (500, 500, 800, 700)
        caja = compute_crop_box(bbox, context_padding_for(bbox), (2000, 2000))

        assert caja[0] <= bbox[0] and caja[1] <= bbox[1]
        assert caja[2] >= bbox[2] and caja[3] >= bbox[3]

    def test_el_recorte_sigue_siendo_cuadrado(self) -> None:
        bbox = (100, 100, 400, 200)
        izquierda, arriba, derecha, abajo = compute_crop_box(
            bbox, context_padding_for(bbox), (2000, 2000)
        )

        assert derecha - izquierda == abajo - arriba

    @pytest.mark.parametrize("lado", [64, 512, 4000])
    def test_funciona_en_fotos_de_cualquier_tamaño(self, lado: int) -> None:
        bbox = (lado // 4, lado // 4, lado // 2, lado // 2)
        caja = compute_crop_box(bbox, context_padding_for(bbox), (lado, lado))

        assert caja[2] > caja[0] and caja[3] > caja[1]


class TestElPipelineUsaElContextoProporcional:
    """El calculo no sirve de nada si el pipeline sigue usando el valor fijo."""

    def test_sin_valor_explicito_el_contexto_sale_de_lo_marcado(self) -> None:
        from PIL import Image

        from app.services.inpaint_pipeline import MaskedEditSettings, run_masked_edit

        base = Image.new("RGB", (2000, 2000), (128, 128, 128))
        mascara = Image.new("L", (2000, 2000), 0)
        # Una marca de 300 px, del tamaño de una boca en una foto grande.
        for y in range(1000, 1300):
            for x in range(1000, 1300):
                mascara.putpixel((x, y), 255)

        recibido: dict = {}

        def modelo(image, mask, width, height):
            recibido["lado"] = image.size[0]
            return image

        run_masked_edit(
            base,
            mascara,
            model=modelo,
            settings=MaskedEditSettings(
                dilate_px=8, feather_px=8, padding_px=None, target_side=512
            ),
        )

        assert recibido, "el modelo no se llamo"

    def test_un_valor_explicito_manda_sobre_el_calculo(self) -> None:
        # Quien quiera mas o menos contexto tiene que poder pedirlo: hay marcas
        # que solo necesitan continuar una textura y prefieren el detalle.
        from app.services.inpaint_pipeline import resolve_padding

        assert resolve_padding(200, (0, 0, 300, 300)) == 200

    def test_sin_valor_explicito_se_calcula(self) -> None:
        from app.services.inpaint_pipeline import resolve_padding

        assert resolve_padding(None, (0, 0, 300, 300)) == context_padding_for((0, 0, 300, 300))
