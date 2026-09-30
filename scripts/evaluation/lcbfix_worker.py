"""LCB-only extractor revision; preserves the original grader and limits."""
import re
import internal_worker as original


def extract(text):
    if not text:
        return ''
    match = re.search(r'```(?:python3|python)?\s*\n(.*?)```', text, re.DOTALL)
    return match.group(1).rstrip() if match else text.rstrip()


def main():
    old = original.author_helpers
    def helpers():
        return dict(old(), _extract_code_block=extract)
    original.author_helpers = helpers
    original.main()


if __name__ == '__main__':
    main()
