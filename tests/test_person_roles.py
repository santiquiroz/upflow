import cv2
import numpy as np

from app.services.person_detector import PersonBox
from app.services.person_roles import (
    anchor_point,
    assign_faces,
    classify_roles,
    face_box,
    merge_duplicates,
    role_scores,
)


def scene_with(
    big_sharp: tuple[int, int, int, int],
    small_blur: tuple[int, int, int, int],
) -> np.ndarray:
    rgb = np.full((2200, 1500, 3), 128, dtype=np.uint8)
    x1, y1, x2, y2 = big_sharp
    pattern = (np.indices((y2 - y1, x2 - x1)).sum(axis=0) % 2) * 176 + 40
    rgb[y1:y2, x1:x2] = pattern[:, :, None]

    x1, y1, x2, y2 = small_blur
    patch = np.full((y2 - y1, x2 - x1, 3), 128, dtype=np.uint8)
    patch[::4, ::4] = 140
    rgb[y1:y2, x1:x2] = cv2.GaussianBlur(patch, (0, 0), sigmaX=8)
    return rgb


def test_big_sharp_centered_person_is_subject_small_blurred_edge_is_background():
    rgb = scene_with(big_sharp=(800, 200, 1400, 1900), small_blur=(50, 700, 200, 1000))
    roles = classify_roles(
        rgb,
        [
            PersonBox((50, 700, 200, 1000), 0.8),
            PersonBox((800, 200, 1400, 1900), 0.9),
        ],
    )
    assert roles == ["background", "subject"]


def test_all_similar_people_are_all_subjects():
    rgb = np.full((600, 900, 3), 128, dtype=np.uint8)
    people = [
        PersonBox((50, 100, 250, 500), 0.9),
        PersonBox((350, 100, 550, 500), 0.9),
        PersonBox((650, 100, 850, 500), 0.9),
    ]
    assert set(classify_roles(rgb, people)) == {"subject"}


def test_anchor_point_uses_face_center_or_upper_quarter():
    face_anchor = anchor_point((0, 0, 100, 400), (20, 10, 60, 50))
    box_anchor = anchor_point((0, 0, 100, 400), None)
    assert face_anchor == (40, 30)
    assert box_anchor == (50, 50)
    assert all(type(value) is int for value in face_anchor)
    assert all(type(value) is int for value in box_anchor)


def test_face_box_none_on_blank():
    assert face_box(np.zeros((400, 300, 3), np.uint8), (0, 0, 300, 400)) is None


def test_face_box_selects_largest_face_in_upper_half_and_returns_python_ints(
    monkeypatch,
):
    class FakeCascade:
        region_shape = None

        def detectMultiScale(self, region, *_args, **_kwargs):
            self.region_shape = region.shape
            return np.array([[5, 10, 20, 20], [40, 15, 30, 40]])

    cascade = FakeCascade()
    monkeypatch.setattr("app.services.person_roles.get_face_cascade", lambda: cascade)
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)

    result = face_box(rgb, (30, 20, 150, 180))

    assert result == (70, 35, 100, 75)
    assert cascade.region_shape == (80, 120)
    assert all(type(value) is int for value in result)


def _fixed_faces(faces):
    return lambda rgb, box: list(faces)


def test_face_at_left_edge_of_big_box_goes_to_the_small_box_centred_on_it():
    rgb = np.zeros((400, 500, 3), np.uint8)
    big = (80, 0, 480, 400)
    small = (60, 100, 140, 300)
    face = (80, 110, 120, 150)

    faces = assign_faces(rgb, [big, small], find_faces=_fixed_faces([face]))

    assert faces == [None, face]


def test_two_people_never_share_a_face():
    rgb = np.zeros((400, 500, 3), np.uint8)
    first = (60, 0, 160, 400)
    second = (40, 0, 200, 400)
    face = (95, 50, 125, 90)

    faces = assign_faces(rgb, [first, second], find_faces=_fixed_faces([face]))

    assert faces.count(face) == 1
    assert faces == [face, None]


def test_assign_faces_drops_near_duplicate_detections_of_one_face():
    rgb = np.zeros((400, 500, 3), np.uint8)
    box = (60, 0, 160, 400)
    faces = [(90, 50, 130, 90), (92, 52, 132, 92)]

    assigned = assign_faces(rgb, [box], find_faces=_fixed_faces(faces))

    assert assigned == [(90, 50, 130, 90)]


def test_merge_duplicates_drops_the_contained_same_head_box_and_keeps_max_score():
    subject = PersonBox((3324, 1187, 4205, 3917), 0.7)
    duplicate = PersonBox((3546, 1190, 4068, 2077), 0.9)

    merged = merge_duplicates([duplicate, subject])

    assert merged == [PersonBox((3324, 1187, 4205, 3917), 0.9)]


def test_merge_duplicates_keeps_a_short_person_standing_behind():
    subject = PersonBox((3847, 978, 5988, 3988), 0.8)
    behind = PersonBox((3980, 1304, 4172, 1914), 0.85)

    merged = merge_duplicates([subject, behind])

    assert merged == [subject, behind]


def test_role_scores_uses_compact_laplacian_and_float64_variance(monkeypatch):
    original_laplacian = cv2.Laplacian
    dtypes = []

    def laplacian(gray, dtype):
        dtypes.append(dtype)
        return original_laplacian(gray, dtype)

    monkeypatch.setattr(cv2, "Laplacian", laplacian)
    rgb = np.random.default_rng(42).integers(0, 256, (80, 100, 3), dtype=np.uint8)

    scores = role_scores(rgb, [PersonBox((10, 10, 50, 70), 0.9)])

    assert len(scores) == 1
    assert dtypes == [cv2.CV_16S]
