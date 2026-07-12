# roundtrip_value_test.py
import numpy as np
import roundtrip_value as V

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
