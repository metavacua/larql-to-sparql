# roundtrip_value_test.py
import numpy as np
import struct
import json as _json
import roundtrip_value as V
import roundtrip_diff as D2

def test_decode_f32_roundtrips():
    arr = np.array([1.5, -2.0, 0.0], dtype=np.float32)
    out = V.decode_tensor(arr.tobytes(), "F32")
    assert np.array_equal(out, arr)

def test_decode_bf16_upper16_bits():
    # bf16 of 1.0 is 0x3F80; little-endian bytes 80 3F
    data = np.array([0x3F80, 0xC000], dtype="<u2").tobytes()  # 1.0, -2.0
    out = V.decode_tensor(data, "BF16")
    assert out.dtype == np.float32
    assert np.allclose(out, [1.0, -2.0])

def test_decode_unsupported_raises():
    import pytest
    with pytest.raises(ValueError):
        V.decode_tensor(b"\x00", "Q4_K")

def test_tensor_value_metrics_exact_and_close():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert V.tensor_value_metrics(a, a) == {
        "comparable": True, "n_total": 3, "n_differing": 0,
        "max_abs_diff": 0.0, "l2": 0.0}
    b = np.array([1.0, 2.0, 3.5], dtype=np.float32)
    m = V.tensor_value_metrics(a, b)
    assert m["n_differing"] == 1
    assert abs(m["max_abs_diff"] - 0.5) < 1e-6

def test_tensor_value_metrics_shape_mismatch():
    a = np.zeros(3, dtype=np.float32); b = np.zeros(4, dtype=np.float32)
    m = V.tensor_value_metrics(a, b)
    assert m["comparable"] is False
    assert m["shape_a"] == [3] and m["shape_b"] == [4]

def _st(path, tensors):  # tensors: {name: (dtype_str, np.ndarray)}
    header, buf, off = {}, bytearray(), 0
    for name, (dt, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(raw)]}
        buf += raw; off += len(raw)
    blob = _json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob))); f.write(blob); f.write(bytes(buf))

def test_safetensors_value_diff_cross_dtype(tmp_path):
    a = tmp_path / "a.safetensors"; b = tmp_path / "b.safetensors"
    v = np.array([1.0, 2.0], dtype=np.float32)
    _st(str(a), {"w": ("F32", v)})
    # bf16 of [1.0, 2.0] = 0x3F80, 0x4000
    bf = np.array([0x3F80, 0x4000], dtype="<u2")
    _st(str(b), {"w": ("BF16", bf)})
    d = V.safetensors_value_diff(str(a), str(b))
    assert d["w"]["comparable"] is True
    assert d["w"]["max_abs_diff"] == 0.0   # 1.0/2.0 exactly representable in bf16
    assert d["_bytes_equal"] is False       # F32 bytes != BF16 bytes

def test_safetensors_value_diff_degrades_on_missing_data_offsets(tmp_path):
    a = tmp_path / "a.safetensors"; b = tmp_path / "b.safetensors"
    v = np.array([1.0, 2.0], dtype=np.float32)
    _st(str(a), {"w": ("F32", v)})
    # Hand-build a header whose tensor spec omits data_offsets entirely.
    data = v.tobytes()
    blob = _json.dumps({"w": {"dtype": "F32", "shape": [2]}}).encode("utf-8")
    with open(str(b), "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(data)
    d = V.safetensors_value_diff(str(a), str(b))
    assert d["w"]["comparable"] is False
