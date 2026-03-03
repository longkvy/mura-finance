class LocalCorpus:
    def get_context(self, row: dict) -> list[str]:
        texts = []

        body = row.get("text")
        if isinstance(body, str) and body.strip():
            texts.append(body.strip())

        return texts
