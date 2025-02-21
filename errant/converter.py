def convert_m2_to_text(m2_string: str) -> dict:
    """
    Convert M2 formatted text to corrected sentence.
    
    Args:
        m2_string (str): String in M2 format
        
    Returns:
        dict: Dictionary with original and corrected sentences
    """
    if not m2_string:
        raise ValueError("Input string is empty")
    
    # Split the input into lines
    lines = m2_string.strip().split('\n')
    
    if len(lines) < 2:
        raise ValueError("Input string is not a valid M2 formatted string: {}".format(m2_string))
    
    # Get the original sentence (first line starting with S)
    original = lines[0][2:].strip()  # Remove 'S ' from the start
    
    # Convert the sentence to a list of characters for easier manipulation
    tokens = original.split()
    
    # Store all edits: (start_pos, end_pos, replacement)
    edits = []
    
    # Process annotation lines
    for line in lines[1:]:
        if not line.startswith('A'):
            continue
            
        # Parse the annotation line
        parts = line[2:].split('|||')  # Remove 'A ' from the start
        if len(parts) < 3:
            continue
            
        # Get the position indices and replacement
        try:
            start, end = map(int, parts[0].split())
            replacement = parts[2]
            edits.append((start, end, replacement))
        except (ValueError, IndexError):
            continue
    
    # Sort edits in reverse order (to handle overlapping edits)
    # simply reverse the list instead of sorting to avoid single-point insertions
    # e.g., a sequence of insertions at position (1 1)
    edits = reversed(edits)
    
    # Apply the edits
    for start, end, replacement in edits:
        if start < 0 or end < 0:
            continue
        if replacement == '':
            tokens = tokens[:start] + tokens[end:]
        else:
            tokens = tokens[:start] + [replacement] + tokens[end:]

    corrected = ' '.join(tokens).strip()
    
    return {
        'original': original,
        'corrected': corrected
    }

def convert_m2_file(file_path: str) -> list[dict]:
    """
    Convert an M2 file to a list of dictionaries with original and corrected sentences.
    
    Args:
        file_path (str): Path to the M2 file
        
    Returns:
        list: List of dictionaries with original and corrected sentences
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content into sentence blocks
    blocks = content.strip().split('\n\n')
    
    # Convert each block
    results = []
    for block in blocks:
        if block.strip():
            results.append(convert_m2_to_text(block))
    
    return results
