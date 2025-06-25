import os


def generate_file_tree(startpath, output_file='PROJECT_STRUCTURE.md'):
    with open(output_file, 'w') as f:
        f.write(f"# Project Structure for `{startpath}`\n\n")

        for root, dirs, files in os.walk(startpath):
            # Calculate indentation level based on folder depth
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * level

            folder_name = os.path.basename(root) if os.path.basename(root) else root
            f.write(f"{indent}- **{folder_name}/**\n")

            subindent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f"{subindent}- {file}\n")


if __name__ == "__main__":
    project_dir = "./DHIS2_CHAT_DEMO"  # Change this to your project root directory
    generate_file_tree(project_dir)
    print("Project structure has been written to PROJECT_STRUCTURE.md")
