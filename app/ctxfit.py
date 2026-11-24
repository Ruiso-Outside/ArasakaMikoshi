"""Faz uma tarefa longa com ferramentas (Codex, agentes) caber no contexto do modelo.

Ler um arquivo grande ou rodar muitos passos passava do contexto e o pedido falhava. Aqui os resultados antigos so sao
encurtados quando falta espaco, e de uma vez ate sobrar folga: encurtar um pouco a cada passo mudaria o comeco da
conversa toda hora e faria o motor reler tudo. A conversa salva nao muda: so o que vai para o modelo.
"""
import json
import re

KEEP_OLD = 400  # caracteres que ficam de um resultado antigo
OVERFLOW_RE = re.compile(r"exceed\w*\b.{0,40}\bcontext|context (size|length).{0,40}exceed|exceed_context_size", re.I)
OLD_NOTE = "\n... (resultado antigo encurtado para caber no contexto; consulte de novo se precisar)"


def is_overflow(error):
    return bool(OVERFLOW_RE.search(str(error)))


class Fit:
    def __init__(self, n_ctx, tools=None):
        self.n_ctx = int(n_ctx or 0)
        self.base = len(json.dumps(tools or [], ensure_ascii=False))
        self.per_token = 3.0  # caracteres por token; aprendido com as contagens do motor

    # -- tamanhos
    @staticmethod
    def chars(messages):
        n = 0
        for m in messages:
            c = m.get("content")
            n += 30
            if isinstance(c, str):
                n += len(c)
            elif isinstance(c, list):
                n += sum(len(p["text"]) if p.get("type") == "text" else 3000 for p in c if isinstance(p, dict))
            n += sum(len((tc.get("function") or {}).get("arguments") or "") + 40 for tc in m.get("tool_calls") or [])
        return n

    def learn(self, sent_chars, timings):
        """Ajusta a conta com os tokens que o motor leu de verdade."""
        t = timings or {}
        tokens = int(t.get("prompt_n") or 0) + int(t.get("cache_n") or 0)
        if tokens > 300:
            self.per_token = min(5.0, max(1.5, (sent_chars + self.base) / tokens))

    def tokens(self, messages):
        return (self.chars(messages) + self.base) / self.per_token

    def reserve(self):
        """Espaco deixado para a resposta."""
        return min(4096, max(1024, self.n_ctx // 4))

    def result_limit(self):
        """Maior resultado de uma ferramenta, em caracteres: um quarto do contexto."""
        return 60_000 if not self.n_ctx else max(3000, min(60_000, int(self.n_ctx * 0.25 * 3)))

    def cap(self, text):
        limit = self.result_limit()
        if not isinstance(text, str) or len(text) <= limit:
            return text
        head = limit * 2 // 3
        return text[:head] + f"\n\n... ({len(text) - limit} caracteres omitidos para caber no contexto) ...\n\n" + text[-(limit - head):]

    # -- encurtar
    def fit(self, messages, hard=False):
        """Encurta `messages` (na propria lista, sem alterar os objetos) se nao couberem. True se mudou algo.
        hard: o motor ja recusou o pedido; encurta mais, mesmo sem saber o contexto."""
        if not hard and (not self.n_ctx or self.tokens(messages) <= self.n_ctx - self.reserve()):
            return False
        target = (self.n_ctx or 8192) * (0.4 if hard else 0.55)
        recent = next((i for i in range(len(messages) - 1, -1, -1)
                       if messages[i].get("role") == "assistant" and messages[i].get("tool_calls")), len(messages))
        changed = False

        def set_(i, **new):
            nonlocal changed
            messages[i] = {**messages[i], **new}
            changed = True
            return self.tokens(messages) <= target

        for i in range(recent):  # 1) resultados antigos, do mais velho para o mais novo
            c = messages[i].get("content")
            if messages[i].get("role") == "tool" and isinstance(c, str) and len(c) > KEEP_OLD + len(OLD_NOTE) + 100:
                if set_(i, content=c[:KEEP_OLD] + OLD_NOTE):
                    return True
        for i in range(recent):  # 2) arquivos inteiros que o modelo escreveu antes
            calls = messages[i].get("tool_calls")
            if calls and any(len((tc.get("function") or {}).get("arguments") or "") > 1500 for tc in calls):
                if set_(i, tool_calls=[self._short_call(tc) for tc in calls]):
                    return True
        for size in (8000, 4000, 2000):  # 3) os resultados do passo atual
            for i in range(recent, len(messages)):
                c = messages[i].get("content")
                if messages[i].get("role") == "tool" and isinstance(c, str) and len(c) > size + 200:
                    if set_(i, content=c[:size] + "\n... (cortado para caber no contexto; leia o resto em partes menores)"):
                        return True
        for i in range(1, recent):  # 4) respostas antigas
            c = messages[i].get("content")
            if messages[i].get("role") == "assistant" and isinstance(c, str) and len(c) > 1500:
                if set_(i, content=c[:1000] + "\n... (resposta antiga encurtada)"):
                    return True
        return changed

    @staticmethod
    def _short_call(tc):
        fn = dict(tc.get("function") or {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
            if isinstance(args, dict):
                for k, v in args.items():
                    if isinstance(v, str) and len(v) > 500:
                        args[k] = v[:300] + "... (encurtado)"
                fn["arguments"] = json.dumps(args, ensure_ascii=False)
        except ValueError:
            pass
        return {**tc, "function": fn}

    # -- chamar o motor
    def call(self, engine_call, payload, on_shrink=None):
        """engine_call(payload) com as mensagens cabendo no contexto; se o motor recusar por tamanho, encurta e tenta de novo."""
        messages = payload["messages"]
        if self.fit(messages) and on_shrink:
            on_shrink()
        self.sent = self.chars(messages)
        for attempt in range(2):
            try:
                return engine_call(payload)
            except RuntimeError as e:
                if not is_overflow(e):
                    raise
                if attempt or not self.fit(messages, hard=True):
                    raise RuntimeError("A tarefa não cabe no contexto do modelo. Aumente o contexto em Configurações → Modelo e runtime, "
                                       f"peça uma parte menor ou comece uma conversa nova. ({e})") from None
                self.sent = self.chars(messages)
                if on_shrink:
                    on_shrink()
