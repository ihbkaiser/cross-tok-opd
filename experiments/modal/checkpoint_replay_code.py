"""Two explicit extraction contracts for diagnosis of identical raw outputs."""
import re


def extract_code(text,variant):
    if variant=='author':
        pattern=r'```(?:python)?\s*\n(.*?)```'
    elif variant=='python-version-aware':
        pattern=r'```(?:(?:python\d*(?:\.\d+)*)|py)?[^\S\r\n]*\r?\n(.*?)```'
    else:
        raise ValueError('Unknown extraction contract')
    match=re.search(pattern,text,re.S)
    return match.group(1).rstrip() if match else text.rstrip()
