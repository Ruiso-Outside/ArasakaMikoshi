"""Windows: o motor fecha ao abrir quando o Visual C++ do sistema e antigo. Aqui o app descobre isso e explica.

O llama.cpp e compilado com o Visual C++ novo; com o msvcp140.dll anterior a 14.40 os programas fecham com erro de acesso
a memoria (0xC0000005) antes de escrever qualquer linha, e a tela so dizia "O motor fechou".
"""
import os

MIN_VERSION = (14, 40)
ACCESS_VIOLATION, DLL_NOT_FOUND, ENTRY_NOT_FOUND = 0xC0000005, 0xC0000135, 0xC0000139
COMMAND = "winget install --id Microsoft.VCRedist.2015+.x64 -e --force"
PAGE = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def vc_runtime_version():
    """Versao do msvcp140.dll do Windows, ex.: (14, 28); None se nao existir ou nao der para ler."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "msvcp140.dll")
        ver = ctypes.WinDLL("version")
        size = ver.GetFileVersionInfoSizeW(path, None)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        ptr, n = ctypes.c_void_p(), ctypes.c_uint()
        if not ver.GetFileVersionInfoW(path, 0, size, buf) or not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(n)):
            return None
        info = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))  # VS_FIXEDFILEINFO: dwFileVersionMS e o 3o campo
        return info[2] >> 16, info[2] & 0xFFFF
    except (OSError, AttributeError, ValueError):
        return None


def crash_hint(returncode, version="?"):
    """Explicacao para o motor que fechou por causa do Visual C++ ("" se a causa for outra)."""
    if os.name != "nt" and version == "?":
        return ""
    code = (returncode or 0) & 0xFFFFFFFF
    if code not in (ACCESS_VIOLATION, DLL_NOT_FOUND, ENTRY_NOT_FOUND):
        return ""
    v = vc_runtime_version() if version == "?" else version
    if v and v >= MIN_VERSION:
        return ""
    have = f"está na versão {v[0]}.{v[1]}" if v else "não está instalado"
    return (f"O motor não abre porque o Visual C++ do Windows {have}; ele precisa da {MIN_VERSION[0]}.{MIN_VERSION[1]} ou mais nova.\n"
            f"Instale o pacote oficial da Microsoft, feche o Arasaka Mikoshi e abra de novo:\n{PAGE}\n"
            f"Ou, no PowerShell: {COMMAND}")
