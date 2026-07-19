class CharacterVocabulary:
    def __init__(self, text: str):
        self.characters = sorted(set(text))
        self.character_to_id = {
            character: index
            for index, character in enumerate(self.characters)
        }
        self.id_to_character = {
            index: character
            for index, character in enumerate(self.characters)
        }

    def encode(self, text: str) -> list[int]:
        return [self.character_to_id[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.id_to_character[token_id] for token_id in token_ids)
