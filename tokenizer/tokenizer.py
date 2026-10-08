import json
import re


class CharTokenizer:

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, text):
        text = re.sub(r"\s+", " ", text.strip())

        special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ]

        characters = sorted(set(text))
        vocabulary = special_tokens + characters

        self.token_to_id = {
            token: index
            for index, token in enumerate(vocabulary)
        }

        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }

        print(f"Vocabulary size: {len(vocabulary)}")

    def encode(self, text):
        text = re.sub(r"\s+", " ", text.strip())

        return [
            self.token_to_id.get(
                character,
                self.token_to_id["<UNK>"]
            )
            for character in text
        ]

    def decode(self, ids):
        result = []

        for token_id in ids:
            token = self.id_to_token.get(
                token_id,
                "<UNK>"
            )

            if token not in ["<PAD>", "<BOS>", "<EOS>"]:
                result.append(token)

        return "".join(result)

    def save(self, path):
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {
                str(k): v
                for k, v in self.id_to_token.items()
            }
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

        print(f"Tokenizer saved to: {path}")


if __name__ == "__main__":

    tokenizer = CharTokenizer()

    sample_text = """
    Hello! My name is Charan.
    I am building my own language model.
    Artificial intelligence can automate tasks.
    """

    tokenizer.train(sample_text)

    text = "Hello Charan!"

    encoded = tokenizer.encode(text)

    print("\nOriginal:")
    print(text)

    print("\nEncoded:")
    print(encoded)

    decoded = tokenizer.decode(encoded)

    print("\nDecoded:")
    print(decoded)

    tokenizer.save("tokenizer.json")