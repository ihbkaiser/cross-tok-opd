import sys
from pathlib import Path
import ast
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/evaluation'))
from lcbfix_worker import extract
from contract_eval import author_helpers
from lcbfix import read_only


@pytest.mark.parametrize('tag',['python','python3',''])
def test_closed_python_fences(tag):
    assert extract('```'+tag+'\nprint(5)\n```')=='print(5)'


def test_python3_regression_and_no_algorithm_repair():
    raw='```python3\nprint(5)\n```'
    with pytest.raises(SyntaxError):
        ast.parse(author_helpers()['_extract_code_block'](raw))
    ast.parse(extract(raw))
    broken='```python\nfor i in x and for j in y:\n pass\n```'
    with pytest.raises(SyntaxError):
        ast.parse(extract(broken))


@pytest.mark.parametrize('text',['print(5)','```python\nprint(5)\n```','```\nprint(5)\n```',
                                '```python3\nprint(', '```javascript\nlet x=1;\n```'])
def test_legacy_behavior_unchanged_outside_complete_python3(text):
    assert extract(text)==author_helpers()['_extract_code_block'](text)


def test_source_reader_never_repairs_partial_journal(tmp_path):
    path=tmp_path/'responses.jsonl'
    raw='{"id":"a"}\n{"id":'
    path.write_text(raw)
    with pytest.raises(ValueError):
        read_only(path)
    assert path.read_text()==raw
