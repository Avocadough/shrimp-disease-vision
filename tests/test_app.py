import io
import shutil
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
import ultralytics
from PIL import Image
from streamlit.testing.v1 import AppTest


def test_app_starts_and_identifies_itself_as_experimental():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_v2.py"), default_timeout=20).run()
    assert not app.exception
    assert any("not a diagnosis" in caption.value.lower() for caption in app.caption)


@pytest.mark.parametrize("detected", [False, True])
def test_simulated_upload_uses_checkpoint_and_model_labels(tmp_path, monkeypatch, detected):
    script = tmp_path / "app_v2.py"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "app_v2.py", script)
    checkpoint = tmp_path / "runs/detect/shrimp_disease_best/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder for a mocked model")

    image = io.BytesIO()
    original = Image.new("RGB", (4, 4))
    original.putpixel((0, 0), (255, 0, 0))
    original.putpixel((1, 0), (0, 0, 255))
    original.save(image, format="PNG")
    image.seek(0)
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: image)
    monkeypatch.setattr(st, "camera_input", lambda *a, **kw: None)
    monkeypatch.setattr(st, "button", lambda *a, **kw: True)
    loaded = []
    model_inputs = []

    class FakeBox:
        cls = [2]
        conf = [0.9]

    class FakeResult:
        names = {2: "BG"}
        boxes = [FakeBox()] if detected else []

        def plot(self, **kwargs):
            return np.zeros((4, 4, 3), dtype=np.uint8)

    class FakeModel:
        def predict(self, *args, **kwargs):
            model_inputs.append(args[0])
            return [FakeResult()]

    def fake_yolo(path):
        loaded.append(Path(path))
        return FakeModel()

    monkeypatch.setattr(ultralytics, "YOLO", fake_yolo)
    app = AppTest.from_file(str(script), default_timeout=20).run()
    assert not app.exception
    assert loaded == [checkpoint]
    assert isinstance(model_inputs[0], Image.Image)
    assert model_inputs[0].getpixel((0, 0)) == (255, 0, 0)
    assert model_inputs[0].getpixel((1, 0)) == (0, 0, 255)
    if detected:
        assert any("Black Gill Disease" in item.value for item in app.markdown)
    else:
        assert any("does not establish" in item.value for item in app.info)


def test_changed_threshold_or_removed_image_clears_old_results(tmp_path, monkeypatch):
    script = tmp_path / "app_v2.py"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "app_v2.py", script)
    checkpoint = tmp_path / "runs/detect/shrimp_disease_best/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder for a mocked model")
    image = io.BytesIO()
    Image.new("RGB", (4, 4)).save(image, format="PNG")
    image.seek(0)
    state = {"image": image, "threshold": 0.25, "clicked": True}

    class FakeResult:
        names = {}
        boxes = []

        def plot(self, **kwargs):
            return np.zeros((4, 4, 3), dtype=np.uint8)

    class FakeModel:
        def predict(self, *args, **kwargs):
            return [FakeResult()]

    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: state["image"])
    monkeypatch.setattr(st, "camera_input", lambda *a, **kw: None)
    monkeypatch.setattr(st, "slider", lambda *a, **kw: state["threshold"])
    monkeypatch.setattr(st, "button", lambda *a, **kw: state["clicked"])
    monkeypatch.setattr(ultralytics, "YOLO", lambda path: FakeModel())
    app = AppTest.from_file(str(script), default_timeout=20).run()
    assert not app.exception
    assert app.session_state["detection_results"] is not None

    state.update(threshold=0.75, clicked=False)
    app.run()
    assert not app.exception
    assert app.session_state["detection_results"] is None
    assert not any(item.value == "🔍 Detection Results" for item in app.header)

    state.update(clicked=True)
    app.run()
    assert app.session_state["detection_results"] is not None
    state.update(image=None, clicked=False)
    app.run()
    assert not app.exception
    assert app.session_state["detection_results"] is None
    assert app.session_state["original_image"] is None
