"""HTTPS que funciona em mais PCs (so biblioteca padrao).

O Python confere os certificados com a lista de certificados raiz do sistema. No Windows essa lista e completada
sob demanda: o certificado raiz de um site so entra nela quando algum programa do proprio Windows (navegador,
curl.exe, PowerShell) visita um site que precisa dele. Num PC novo, o Python falha com
"CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate" ao baixar o motor ou um modelo.

O que este modulo faz:
- usa a lista do sistema e, se o pacote certifi estiver instalado, tambem a lista dele;
- quando a verificacao falha no Windows, visita o mesmo endereco pelo proprio Windows (curl.exe ou PowerShell,
  que usam o Schannel e baixam a raiz que falta), recria a lista e tenta de novo uma vez;
- se ainda assim falhar (antivirus ou proxy que inspeciona o HTTPS com um certificado que nao esta no Windows),
  devolve uma mensagem que diz o que fazer, em vez do erro cru do OpenSSL.
A verificacao do certificado nunca e desligada.
"""
import os
import ssl
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_lock = threading.Lock()
_ctx = None
_warmed = set()


def _build():
    ctx = ssl.create_default_context()  # lista do sistema (no Windows: as lojas ROOT e CA) e SSL_CERT_FILE
    try:
        import certifi  # opcional: lista da Mozilla, se a pessoa tiver instalado
        ctx.load_verify_locations(certifi.where())
    except Exception:  # noqa: BLE001
        pass
    return ctx


def context():
    global _ctx
    with _lock:
        if _ctx is None:
            _ctx = _build()
        return _ctx


def reset():
    global _ctx
    with _lock:
        _ctx = None


def is_cert_error(e):
    reason = getattr(e, "reason", e)
    return isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(reason)


class CertificateError(urllib.error.URLError):
    """A verificacao do certificado falhou mesmo depois de atualizar a lista do Windows."""

    def __init__(self, url, original):
        super().__init__(original)
        self.url = url
        self.host = urllib.parse.urlparse(url).hostname or url

    def __str__(self):
        return (f"Não foi possível verificar o certificado de segurança de {self.host}. Isso acontece quando um "
                "antivírus ou proxy inspeciona as conexões HTTPS, ou quando o Windows está sem os certificados raiz "
                f"atualizados. Abra https://{self.host} uma vez no navegador e tente de novo; se continuar, rode "
                "\"python -m pip install certifi\" ou libere o app na proteção HTTPS do antivírus.")


def _url_of(req):
    return req.full_url if isinstance(req, urllib.request.Request) else str(req)


def warm_windows_roots(url):
    """Visita o endereco pelo proprio Windows para ele baixar o certificado raiz que falta. True se tentou agora."""
    if os.name != "nt" or not url.lower().startswith("https://"):
        return False
    host = urllib.parse.urlparse(url).hostname or ""
    with _lock:
        if host in _warmed:
            return False
        _warmed.add(host)
    env = dict(os.environ, MIKOSHI_TLS_URL=url)  # o endereco vai por variavel de ambiente, nunca dentro do comando
    attempts = (
        ["curl.exe", "-s", "-L", "-I", "-o", "NUL", "--max-time", "20", url],
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         "try { Invoke-WebRequest -UseBasicParsing -Method Head -TimeoutSec 20 -Uri $env:MIKOSHI_TLS_URL | Out-Null } catch {}"],
    )
    for cmd in attempts:
        try:
            subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=40,
                           creationflags=NO_WINDOW)
            return True
        except (OSError, subprocess.SubprocessError):
            continue
    return True


def opener(*handlers):
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context()), *handlers)


def urlopen(req, timeout=30, handlers=()):
    """Como urllib.request.urlopen, com a lista de certificados acima e uma nova tentativa no Windows."""
    try:
        return opener(*handlers).open(req, timeout=timeout)
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as e:
        if not is_cert_error(e):
            raise
        url = _url_of(req)
        if not warm_windows_roots(url):
            raise CertificateError(url, e.reason) from e
    reset()
    try:
        return opener(*handlers).open(req, timeout=timeout)
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as e:
        if is_cert_error(e):
            raise CertificateError(_url_of(req), e.reason) from e
        raise
