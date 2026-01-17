def decode_patch_bytes(patch_bytes: bytes) -> str:
    """Decode patch bytes robustly.

    Prefers UTF-8, then uses UTF-8 with surrogateescape to losslessly preserve
    any non-UTF-8 bytes. Falls back to latin-1, and finally replaces invalid
    sequences to ensure we never raise.
    """
    try:
        return patch_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Lossless round-trip for arbitrary bytes when re-encoding with utf-8
        # using the same error handler.
        try:
            return patch_bytes.decode("utf-8", errors="surrogateescape")
        except Exception:
            try:
                return patch_bytes.decode("latin-1")
            except Exception:
                return patch_bytes.decode("utf-8", errors="replace")