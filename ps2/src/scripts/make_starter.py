"""Make starter code from solutions."""
import argparse
import os
import re

SRC_EXT = '.py'
TEX_EXT = '-sol.tex'
COMMENT_RE = re.compile(r'\s*#')
MARKER_RE = re.compile(r'# \*\*\*|\\(begin|end){answer}')


def main(args):
    for base_path, _, file_names in os.walk(args.root_dir):
        for file_name in file_names:
            if file_name.endswith(SRC_EXT) or file_name.endswith(TEX_EXT):
                # Read file
                file_path = os.path.join(base_path, file_name)
                with open(file_path, 'r') as fh:
                    lines = fh.readlines()

                # Remove solutions
                filtered_lines = remove_solution(lines)
                with open(file_path, 'w') as fh:
                    fh.writelines(filtered_lines)


def remove_solution(lines):
    """Remove all solution lines from lines."""
    filtered_lines = []
    is_solution = False
    for line in lines:
        is_marker = MARKER_RE.search(line)
        if is_marker:
            is_solution = not is_solution
        elif is_solution and not COMMENT_RE.match(line):
            continue
        filtered_lines.append(line)

    return filtered_lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make starter code from solutions')
    parser.add_argument('--root_dir', default='..', type=str)
    main(parser.parse_args())
