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

    print("processed dataset")


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