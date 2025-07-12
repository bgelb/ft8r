import pytest
from tests.utils import ft8code_bits
from utils import decode77, register_callsign, ihashcall


def decode_msg(msg: str) -> str:
    bits = ft8code_bits(msg)[:77]
    return decode77(bits)


def test_free_text():
    msg = "TNX BOB 73 GL"
    assert decode_msg(msg) == msg


def test_dxpedition():
    msg = "K1ABC RR73; W9XYZ <KH1/KH7Z> -08"
    register_callsign("KH1/KH7Z")
    assert decode_msg(msg) == msg


def test_field_day():
    msg = "W9XYZ K1ABC R 17B EMA"
    assert decode_msg(msg) == msg


def test_telemetry():
    msg = "123456789ABCDEF012"
    assert decode_msg(msg) == msg


def test_standard():
    msg = "CQ K1ABC FN42"
    assert decode_msg(msg) == msg


def test_nonstandard():
    msg = "PJ4/K1ABC <W9XYZ>"
    register_callsign("W9XYZ")
    assert decode_msg(msg) == msg


def test_rtty_roundup():
    msg = "TU; KA0DEF K1ABC R 569 MA"
    assert decode_msg(msg) == msg

