
def process_default(
    input_file_path: str,
    output_file_path: str,
    vocab_file=None) -> list:
    """ 
    Does Nothing. Just loads the data into
    preprocessed file
    """
    with open(input_file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.read()

    # Write the processed text to the output file
    with open(output_file_path, 'w') as f:
        f.write(lines)

    print("processed dataset")