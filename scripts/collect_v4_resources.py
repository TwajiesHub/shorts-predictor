#!/usr/bin/env python3
"""Rebuild the small, pinned VADER valence resource; never needed at inference."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT/'data/lexicons'
    manifest = json.loads((directory/'vader_source.json').read_text())
    for name, source in manifest['files'].items():
        with urlopen(source['url'], timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != source['sha256']:
            raise ValueError(f'Source integrity check failed: {name}')
        if name.endswith('vader_lexicon.txt'):
            values = {line.split('\t')[0]: float(line.split('\t')[1])
                      for line in data.decode().splitlines() if '\t' in line}
            (directory/'vader_valence.json').write_text(json.dumps(values, ensure_ascii=False, sort_keys=True)+'\n')
            print(f'Collected {len(values)} sentiment terms; source checksum verified')
        else:
            (directory/'LICENSE.vader.txt').write_bytes(data)


if __name__ == '__main__':
    main()
