import re, os

def replace_includes(markdown_file):
    """Replace include statements with actual file contents."""

    include_pattern = re.compile(r'{%\s*include\s+([^\s]+)\s*%}')  # Match {% include filename.md %}

    with open(markdown_file, 'r', encoding='utf-8') as file:
        content = file.read()

    def include_replacer(match):
        """Reads and returns the content of the included file."""
        include_filename = match.group(1).strip()
        if os.path.exists(include_filename):
            with open(include_filename, 'r', encoding='utf-8') as inc_file:
                return inc_file.read()
        else:
            return f"<!-- ERROR: {include_filename} not found -->"

    # Replace include statements
    updated_content = include_pattern.sub(include_replacer, content)

    # Write the processed content back to a new file
    output_file = f"{os.path.splitext(markdown_file)[0]}"
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(updated_content)

    print(f"Processed Markdown file saved as: {output_file}")

if __name__ == "__main__":
    markdown_file = "README.md.tmp"
    replace_includes(markdown_file)
