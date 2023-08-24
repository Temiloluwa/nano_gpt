import os
import json
import torch
from typing import List, Union


def read_file(file_path: str) -> list:
    """Reads the input file and returns the list of lines.
    """
    with open(file_path, 'r') as f:
        lines = f.read()

    return lines


def process_data(input_file_path, 
                processed_file_path,
                dataset_name, 
                vocab_file=None):


    def process_kjv(
        input_file_path: str,
        output_file_path: str,
        vocab_file=None) -> list:
        """ 
        Removes verse reference from input file 
        and writes the processed text to the output file.
        """
        with open(input_file_path, 'r') as f:
            lines = f.readlines()[1:] # Skip the first line which is the header

        processed_lines = []
        for line in lines:
            # Split each line into verse reference and text content
            parts = line.split(' ', 1)
            if len(parts) == 2:
                # Append only the text content to the processed_lines list
                processed_lines.append(parts[1])

        # Write the processed text to the output file
        with open(output_file_path, 'w') as f:
            preprocessed_text = ''.join(processed_lines)
            f.write(preprocessed_text)

    if dataset_name == "kjv":
        process_fn = process_kjv

    process_fn(input_file_path, processed_file_path, vocab_file)
    print(f'processed file saved @ {processed_file_path}')



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
    