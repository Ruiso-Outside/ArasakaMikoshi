"""Mantem o comeco da conversa igual entre um pedido e o outro, para o motor reaproveitar o cache do prompt.

Os trechos da base de conhecimento, da pesquisa na web e da documentacao do Construct entram na ultima mensagem do
usuario. No pedido seguinte eles sumiam dessa mensagem (ou mudavam o prompt do sistema) e o motor relia a conversa dali
em diante; nos modelos com janela deslizante ou hibridos, muitas vezes desde o comeco. Aqui o servidor lembra, so na
memoria, o texto que cada mensagem levou e o repete enquanto couber no contexto.
"""
import hashlib
import json
import threading
from collections import OrderedDict

MAX_ITEMS, MAX_CHARS = 300, 6_000_000
CHARS_PER_TOKEN = 3


def budget(n_ctx, share=0.5):
    """Caracteres que podem ser gastos com o historico (aproximado)."""
    return int((n_ctx or 8192) * CHARS_PER_TOKEN * share)


def text_len(messages):
    """Tamanho aproximado das mensagens, sem contar imagens e audio."""
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            n += sum(len(p.get("text") or "") for p in c if isinstance(p, dict))
        n += sum(len((tc.get("function") or {}).get("arguments") or "") for tc in m.get("tool_calls") or [])
    return n


def prepend(messages, text):
    """Copia das mensagens com `text` no comeco da ultima mensagem do usuario."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            if isinstance(m["content"], list):
                m["content"] = [dict(x) for x in m["content"]]
                for part in m["content"]:
                    if part.get("type") == "text":
                        part["text"] = text + part["text"]
                        break
            else:
                m["content"] = text + (m["content"] or "")
            break
    return out


class ContextMemory:
    def __init__(self):
        self.items = OrderedDict()
        self.chars = 0
        self.lock = threading.Lock()

    @staticmethod
    def _keys(messages):
        """Uma chave por mensagem, que depende de tudo o que veio antes dela (menos o prompt do sistema)."""
        h, keys = hashlib.sha256(), []
        for m in messages:
            if m.get("role") != "system":
                h.update(json.dumps([m.get("role"), m.get("content"), m.get("tool_calls")], ensure_ascii=False, sort_keys=True).encode("utf-8"))
            keys.append(h.hexdigest())
        return keys

    def stabilize(self, original, augmented, n_ctx):
        """original: mensagens como vieram; augmented: com os trechos na ultima mensagem do usuario.
        Devolve `augmented` com as mensagens antigas do usuario como foram enviadas da primeira vez, se couberem."""
        if len(original) != len(augmented):
            return augmented
        keys = self._keys(original)
        last = next((i for i in range(len(original) - 1, -1, -1) if original[i].get("role") == "user"), None)
        if last is None:
            return augmented
        with self.lock:
            new, old = augmented[last].get("content"), original[last].get("content")
            if isinstance(new, str) and isinstance(old, str) and new != old:
                self.chars -= len(self.items.pop(keys[last], ""))
                self.items[keys[last]] = new
                self.chars += len(new)
                while len(self.items) > MAX_ITEMS or self.chars > MAX_CHARS:
                    self.chars -= len(self.items.popitem(last=False)[1])
            out = list(augmented)
            for i in range(last):
                if original[i].get("role") == "user" and keys[i] in self.items:
                    self.items.move_to_end(keys[i])
                    out[i] = {**out[i], "content": self.items[keys[i]]}
        # tudo ou nada: guardar so uma parte mudaria o comeco da conversa de novo
        return out if text_len(out) <= budget(n_ctx, 0.6) else augmented


MEMORY = ContextMemory()
