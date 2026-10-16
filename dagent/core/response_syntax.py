"""Narrow, auditable decoding of literal backslashes in JSON string values.

No completion of truncated objects, inferred fields, structural repairs, or model
calls. Invalid backslash escapes are made literal only in string VALUES. Legal
JSON escapes are never changed, including their existing decoded semantics.
"""
import json

VERSION = 'json_escape_repair'


def parse(text):
    original = text.strip()
    if original.startswith('```'):
        original = original.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        value = json.loads(original)
        offsets = []
    except json.JSONDecodeError as exc:
        if exc.msg != 'Invalid \\escape':
            raise
        out, offsets = [], []
        i, in_string, string_offsets = 0, False, []
        while i < len(original):
            char = original[i]
            if not in_string:
                out.append(char)
                if char == '"':
                    in_string, string_offsets = True, []
                i += 1
                continue
            if char == '"':
                # Never normalize a key: its identity defines the protocol.
                following = original[i + 1:].lstrip()
                if string_offsets and following.startswith(':'):
                    raise ValueError('Invalid escape in JSON key')
                out.append(char)
                in_string = False
                i += 1
                continue
            if char == '\\':
                if i + 1 >= len(original):
                    raise ValueError('Truncated JSON escape')
                next_char = original[i + 1]
                if next_char not in '"\\/bfnrtu':
                    if ord(next_char) < 32:
                        raise ValueError('Unescaped control character in JSON string')
                    out.append('\\')
                    offsets.append(i)
                    string_offsets.append(i)
                out.extend((char, next_char))
                i += 2
                continue
            out.append(char)
            i += 1
        # Unclosed strings, invalid unicode escapes, commas, and all other
        # syntax errors still fail the ordinary strict decoder.
        value = json.loads(''.join(out))
    if not isinstance(value, dict):
        raise ValueError('API response must be a JSON object')
    return value, dict(policy=VERSION, invalid_escape_offsets=offsets,
                       literal_backslashes_preserved=len(offsets), extra_api_calls=0,
                       structure_inferred=False)
