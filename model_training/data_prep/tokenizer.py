import os
import json
from typing import List, Union


class Tokenizer:
    
    def __init__(self,
                processed_data_path,
                vocab_file):
        self.vocabulary = self.build_vocabulary(
                        processed_data_path, 
                        vocab_file)
        
    def build_vocabulary(
                        self,
                        processed_data_path: str,
                        vocab_file: str):
        """Builds a vocabulary

        Args:
            documents (list): _description_
            character_level (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        def read_file(file_path: str) -> list:
            """Reads the input file and returns the list of lines.
            """
            with open(file_path, 'r') as f:
                lines = f.read()

            return lines

        if os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                self.string_to_token_mapper = json.load(f)
                vocabulary = list(self.string_to_token_mapper.keys())
        else:
            vocabulary = sorted(list(set(read_file(processed_data_path))))
            self.string_to_token_mapper = {string_: token for token, string_ in enumerate(vocabulary)}

        self.token_to_string_mapper = {token: string_ for token, string_ in enumerate(vocabulary)}

        with open(vocab_file, 'w') as f:
            json.dump(self.string_to_token_mapper, f, indent=4)

        print(f'Vocabulary: {"".join(vocabulary)}')
        print(f'Vocabulary size: {len(vocabulary)}')

        return vocabulary


    def encode(self, words: str) -> Union[list,str]:

        """Tokenizes the input strings.

        Args:
            inputs (list): _description_

        Returns:
            _type_: _description_
        """
        mapper = self.string_to_token_mapper        
        mapping = [mapper[ch] for ch in words]

        return mapping

    
    def decode(self, tokens: List[int]):
        """
            Decode a list of tokens
        """            
        mapper = self.token_to_string_mapper
        mapping = "".join([mapper[ch] for ch in tokens])

        return mapping
    