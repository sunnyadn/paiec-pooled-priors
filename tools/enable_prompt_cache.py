#!/usr/bin/env python3
"""Enable explicit 5-minute cache breakpoints without changing prompt text.
No credentials, model execution, or network. Used by make_zip and TINIT builds.
"""
import argparse
import ast
from pathlib import Path

PRIOR_USER = '''head = (_ANCHOR_TXT + "\\n\\nNow the benchmark to estimate:\\n\\n") if USE_ANCHORS else ""
tail = body + "\\n\\n" + desc + "\\n\\nReply with only one number from 0 to 100: the percentage of this benchmark's items this system answers correctly."
user = ([{"type": "text", "text": head, "cache_control": {"type": "ephemeral"}}] if head else []) + [{"type": "text", "text": tail}]
'''

def enable(source):
    tree = ast.parse(source)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if '_call_prior' not in funcs or '_call' not in funcs:
        raise ValueError('Unrecognized predictor: review its reusable prefixes before paid packaging')
    prior = funcs['_call_prior']
    psrc = ast.get_source_segment(source, prior)
    if 'cache_control' not in psrc:
        users = [n for n in prior.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'user' for t in n.targets)]
        if len(users) != 1:
            raise ValueError('Expected one prior user prompt construction')
        # Refuse silent semantic changes when a future prompt differs.
        expected = ast.parse('user = ((_ANCHOR_TXT + "\\n\\nNow the benchmark to estimate:\\n\\n") if USE_ANCHORS else "") + body + "\\n\\n" + desc + "\\n\\nReply with only one number from 0 to 100: the percentage of this benchmark\'s items this system answers correctly."').body[0]
        if ast.dump(users[0], include_attributes=False) != ast.dump(expected, include_attributes=False):
            raise ValueError('Prior prompt changed: review cache split rather than replacing it')
        i = prior.body.index(users[0]); prior.body[i:i+1] = ast.parse(PRIOR_USER).body
    rate = funcs['_call']
    found = False
    for n in ast.walk(rate):
        if isinstance(n, ast.Dict):
            for i,k in enumerate(n.keys):
                if isinstance(k, ast.Constant) and k.value == 'system':
                    value = n.values[i]
                    if ast.unparse(value) == 'RATE_PROMPT.strip()':
                        n.values[i] = ast.parse('[{"type":"text", "text":RATE_PROMPT.strip(), "cache_control":{"type":"ephemeral"}}]', mode='eval').body
                        found = True
                    elif 'cache_control' in ast.unparse(value):
                        found = True
                    else: raise ValueError('Unknown rating system prompt')
    if not found: raise ValueError('Missing Anthropic rating system prompt')
    lines = source.splitlines(keepends=True)
    for node in sorted([prior, rate], key=lambda n:n.lineno, reverse=True):
        lines[node.lineno-1:node.end_lineno] = [ast.unparse(ast.fix_missing_locations(node))+'\n']
    result = ''.join(lines)
    ast.parse(result)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('source',type=Path); ap.add_argument('output',type=Path)
    a=ap.parse_args(); result=enable(a.source.read_text()); a.output.write_text(result)
    print('Prompt caching enabled: fixed prior examples + rating system prefix (short prefixes may not reach provider minimum).')
if __name__ == '__main__': main()
